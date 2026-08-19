"""Contract tests for the frozen B5 full-training/evaluation protocol."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "paper0/manifests/phase3_b5_full_training_evaluation_85604.json"
)
PROTOCOL = (
    ROOT / "paper0/protocol/PHASE3_B5_FULL_TRAINING_EVALUATION_PROTOCOL.md"
)


def load() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_B5_full_scope_authorizes_one_seed_one_step_only() -> None:
    record = load()
    assert record["development_run"] == "85604"
    assert record["sequestered_run"] == "85606"
    assert record["held_out_85606_access_allowed"] is False
    assert set(record["authorized_scope"]) == {
        "B5_seed1701_full_training_on_85604_training_targets",
        "B5_seed1701_checkpoint_selection_on_85604_validation_denoising_loss",
        "one_four_target_M32_evaluator_smoke",
        "one_full_126_target_M32_one_step_85604_validation_evaluation",
        "one_prospective_B5_one_seed_acceptance_gate",
    }
    assert {
        "architecture_or_schedule_sweep",
        "additional_model_seeds",
        "O3_fixed_block_forecast",
        "O4_autonomous_rollout",
        "assimilation",
        "diagnostic_ranking",
        "85606_access",
    } <= set(record["forbidden_scope"])


def test_B5_full_evidence_locks_smoke_parent_and_truth_separated_means() -> None:
    locks = load()["evidence_locks"]
    smoke = locks["B5_smoke"]
    tracked = ROOT / smoke["tracked_result"]
    assert smoke["job_id"] == "6901469"
    assert smoke["passed"] is True and smoke["scientific_result"] is False
    assert sha256(tracked) == smoke["tracked_result_sha256"]
    assert locks["residual_audit"]["sha256"] == (
        "d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67"
    )
    assert locks["H1_training_forecast"]["sha256"] == (
        "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18"
    )
    assert locks["H1_training_forecast"]["truth_separated"] is True
    assert locks["H1_validation_forecast"]["sha256"] == (
        "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
    )
    assert locks["H1_validation_forecast"]["truth_separated"] is True
    assert locks["H1_checkpoint"]["trainable"] is False
    assert locks["H1_checkpoint"]["reselection_allowed"] is False


def test_B5_full_data_is_complete_physical_state_without_time_or_leakage() -> None:
    data = load()["data"]
    assert data["fields"] == ["Ne", "Pe", "Pi", "phi", "Vi"]
    assert data["training_targets"] == [2, 432]
    assert data["guard_frames"] == [432, 496]
    assert data["validation_targets"] == [498, 624]
    assert data["volume_shape"] == [5, 64, 32, 88]
    assert data["zperiod"] == 5 and data["mode_mapping"] == "n=5k"
    assert data["toroidal_roll_augmentation"] is False
    assert data["temporal_windows_are_independent_shots"] is False
    for key in (
        "absolute_time_input_allowed",
        "normalized_frame_index_input_allowed",
        "shot_label_input_allowed",
        "diagnostic_input_allowed",
        "region_mask_input_allowed",
        "future_truth_condition_allowed",
        "guard_frames_read_allowed",
    ):
        assert data[key] is False


def test_B5_full_model_is_fresh_joint_field_EDM_without_physics_loss() -> None:
    record = load()
    model = record["model"]
    assert model["name"] == "B5-H1-JOINT-FIELD-EDM-UNET3D-MINI"
    assert model["initialization"] == "fresh_seed1701_not_smoke_checkpoint"
    assert model["parameter_count"] == 11604709
    assert model["joint_output_fields"] == 5
    assert model["dynamic_condition_channels"] == 10
    assert model["padding_by_axis"] == ["zeros", "zeros", "circular"]
    assert model["full_field"] is True
    assert model["DCAE_or_latent_representation_used"] is False
    assert model["deterministic_parent_trainable"] is False
    assert model["physics_derived_training_loss_allowed"] is False
    edm = record["edm"]
    assert edm["sigma_data"] == 1.0
    assert edm["P_mean"] == -1.2 and edm["P_std"] == 1.2
    assert edm["sampler_steps"] == 18
    assert edm["network_evaluations_per_member"] == 35
    assert edm["stochastic_churn"] == 0.0


def test_B5_full_order_and_budget_are_exact_and_byte_locked() -> None:
    config = load()["full_training"]
    assert config["authorized"] is True
    assert config["seed"] == 1701
    assert config["epochs"] == 100
    assert config["target_presentations"] == 43000
    assert config["microbatch_targets"] == 1
    assert config["gradient_accumulation_targets"] == 4
    assert config["final_partial_accumulation_targets_per_epoch"] == 2
    assert config["optimizer_steps_per_epoch"] == 108
    assert config["total_optimizer_steps"] == 10800
    assert config["EMA_decay_per_optimizer_step"] == 0.999
    assert config["EMA_update"] == (
        "ema_parameter=0.999*ema_parameter+0.001*raw_parameter_after_each_optimizer_step"
    )
    assert config["early_stopping"] is False
    generator = np.random.Generator(np.random.PCG64(67501))
    order = np.ascontiguousarray(
        np.stack(
            [
                generator.permutation(np.arange(2, 432, dtype=np.int64))
                for _ in range(100)
            ]
        ),
        dtype=np.int64,
    )
    assert order.shape == (100, 430)
    assert hashlib.sha256(order.tobytes(order="C")).hexdigest() == config[
        "training_order_raw_sha256"
    ]
    buffer = io.BytesIO()
    np.save(buffer, order, allow_pickle=False)
    assert hashlib.sha256(buffer.getvalue()).hexdigest() == config[
        "training_order_npy_sha256"
    ]
    assert order[0, :12].tolist() == [
        285,
        245,
        370,
        398,
        307,
        179,
        86,
        413,
        252,
        157,
        324,
        358,
    ]


def seed_bank(seed: int, shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
    generator = np.random.Generator(np.random.PCG64(seed))
    values = np.ascontiguousarray(
        generator.integers(
            0,
            np.iinfo(np.uint64).max,
            size=shape,
            dtype=np.uint64,
        ),
        dtype=np.uint64,
    )
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return values, hashlib.sha256(buffer.getvalue()).hexdigest()


def test_B5_full_selection_bank_and_rule_are_exact() -> None:
    selection = load()["checkpoint_selection"]
    assert selection["candidate_completed_epochs"] == list(range(5, 101, 5))
    assert selection["validation_probe_draws_per_target"] == 4
    assert selection["validation_probe_count_per_candidate"] == 504
    assert selection["validation_precision"] == "float32_no_autocast_TF32_disabled"
    values, npy_hash = seed_bank(67503, (126, 4))
    assert hashlib.sha256(values.tobytes(order="C")).hexdigest() == selection[
        "seed_bank_raw_sha256"
    ]
    assert npy_hash == selection["seed_bank_npy_sha256"]
    assert selection["rule"] == (
        "earliest_numerically_lowest_metric_after_complete_100_epoch_budget"
    )
    assert selection["reload_probe"] == {
        "purpose": "mechanical_selected_checkpoint_serialization_check_only",
        "seed": 67504,
        "validation_target_frame": 498,
        "precision": "float32_no_autocast_TF32_disabled",
        "bitwise_exact_output_required": True,
        "checkpoint_selection_allowed": False,
    }
    assert selection["physics_metric_allowed"] is False
    assert selection["sampled_forecast_metric_allowed"] is False
    assert selection["85606_value_allowed"] is False


def test_B5_full_scientific_M32_bank_axes_and_truth_separation_are_exact() -> None:
    forecast = load()["scientific_forecast"]
    values, npy_hash = seed_bank(67532, (126, 32))
    assert hashlib.sha256(values.tobytes(order="C")).hexdigest() == forecast[
        "seed_bank_raw_sha256"
    ]
    assert npy_hash == forecast["seed_bank_npy_sha256"]
    assert forecast["seed_bank_independent_of_checkpoint_selection"] is True
    assert forecast["canonical_shape"] == [126, 32, 1, 5, 64, 32, 88]
    assert forecast["float32_uncompressed_bytes"] == 14533263360
    assert forecast["stored_member_prefixes"] == [4, 8, 16, 32]
    assert forecast["truth_separated_generation"] is True
    for key in (
        "recentring_allowed",
        "inflation_allowed",
        "clipping_allowed",
        "member_rejection_allowed",
        "member_sorting_allowed",
        "regeneration_allowed",
    ):
        assert forecast[key] is False


def test_B5_full_gate_reuses_strict_joint_and_transport_thresholds() -> None:
    record = load()
    gate = record["acceptance"]
    assert sha256(ROOT / gate["source_protocol"]) == gate["source_protocol_sha256"]
    assert gate["field_mean_error_relative_to_H1_maximum"] == 1.05
    assert gate["field_spread_skill_primary_range"] == [0.8, 1.25]
    assert gate["material_power_ratio_range"] == [0.75, 1.3]
    assert gate["material_realization_coherence_minimum"] == 0.8
    assert gate["material_cross_phase_error_degrees_maximum"] == 20.0
    assert gate["separatrix_transport_relative_L2_maximum"] == 0.3
    assert gate["separatrix_transport_correlation_minimum"] == 0.8
    assert gate["Monte_Carlo_M16_vs_M32_relative_tolerance"] == 0.1
    assert gate["temporal_blocks_required"] == 5
    assert gate["all_integrity_field_joint_transport_and_Monte_Carlo_families_required"] is True
    post = record["post_gate"]
    assert post["O3_protocol_may_be_written_only_if_gate_passes"] is True
    assert post["O3_launch_authorized"] is False
    assert post["85606_access_authorized"] is False
    assert post["assimilation_authorized"] is False
    assert post["diagnostic_ranking_authorized"] is False


def test_B5_full_protocol_states_data_limit_selection_and_claim_boundary() -> None:
    text = " ".join(PROTOCOL.read_text(encoding="utf-8").split())
    for phrase in (
        "430 adjacent target states from one simulation",
        "Repeated corruptions are not new independent plasma realizations",
        "The non-scientific smoke checkpoint is not a warm start",
        "No validation statistic changes these scales",
        "early stopping: prohibited",
        "select the earliest candidate",
        "forecast closes and hashes before any target truth reader",
        "Nonlinear quantities are later computed for each member",
        "A one-step pass would authorize an O3/O4 protocol",
        "It would not establish autonomous rollout",
    ):
        assert phrase in text
