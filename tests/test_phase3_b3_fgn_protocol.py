"""Regression locks for the prospective Phase 3 B3 FGN protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper0/protocol/PHASE3_B3_FGN_PROTOCOL.md"
MANIFEST = ROOT / "paper0/manifests/phase3_b3_fgn_85604.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_phase3_b3_protocol_and_manifest_are_hash_locked() -> None:
    record = _manifest()
    assert _sha256(PROTOCOL) == (
        "c31d24d843050c5708440018217e48ca66f5c3a3f4ee0ddb110a3287a73292d1"
    )
    assert _sha256(MANIFEST) == (
        "8789c1a922bbeb9817144344563107f6e72a7a7c549436ae698155d93daba900"
    )
    assert record["protocol"] == {
        "path": "paper0/protocol/PHASE3_B3_FGN_PROTOCOL.md",
        "sha256": _sha256(PROTOCOL),
    }


def test_phase3_b3_scope_is_smoke_only_and_keeps_85606_closed() -> None:
    record = _manifest()
    assert record["development_run"] == "85604"
    assert record["sequestered_run"] == "85606"
    assert record["held_out_85606_access_allowed"] is False
    assert record["full_training_authorized"] is False
    assert record["authorized_scope"] == [
        "B3_FGN_H1_implementation",
        "B3_FGN_H1_CPU_tests",
        "B3_FGN_H1_single_seed_bounded_GPU_smoke_85604",
    ]
    assert "B3_full_training_before_new_authorization" in record["forbidden_scope"]
    assert "85606_access" in record["forbidden_scope"]
    smoke = record["implementation_gate"]["gpu_smoke"]
    assert smoke["scientific_result"] is False
    assert smoke["seed"] == 1701
    assert smoke["epochs"] == 2
    assert smoke["training_target_count"] == 16
    assert smoke["validation_target_count"] == 4
    assert smoke["ensemble_members"] == 2
    assert smoke["wandb_online_required"] is True


def test_phase3_b3_uses_frozen_C5P_data_without_time_or_future_truth() -> None:
    record = _manifest()
    data = record["data"]
    assert data["fields"] == ["Ne", "Pe", "Pi", "phi", "Vi"]
    assert data["input_channels"] == "physically_valid_complete_C5P_state"
    assert data["training_targets"] == [2, 432]
    assert data["guard_frames"] == [432, 496]
    assert data["validation_targets"] == [498, 624]
    assert data["cadence_microseconds"] == 3.131905426352636
    assert data["zperiod"] == 5
    assert data["mode_mapping"] == "n=5k"
    for flag in (
        "absolute_time_input_allowed",
        "normalized_frame_index_input_allowed",
        "shot_label_input_allowed",
        "future_truth_input_allowed",
        "guard_frames_read_allowed",
    ):
        assert data[flag] is False


def test_phase3_b3_parent_codec_and_normalization_are_exactly_locked() -> None:
    record = _manifest()
    parent = record["deterministic_parent"]
    assert parent["arm"] == "C5P-H1"
    assert parent["seed"] == 1701
    assert parent["checkpoint_sha256"] == (
        "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
    )
    assert parent["kind"] == "selected_O2_transition"
    assert parent["selected_epoch_zero_based"] == 193
    assert parent["selected_optimizer_step"] == 5238
    assert parent["unexpected_parent_keys_allowed"] is False
    assert parent["noise_disabled_preoptimization_bitwise_identity_required"] is True

    codec = record["codec"]
    assert codec["name"] == "C5P-dcae_l10"
    assert codec["checkpoint_sha256"] == (
        "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
    )
    assert codec["latent_normalization_sha256"] == (
        "afcb0eda5d611d58f6eb2340aa55cfecd1a231b83a6912d9db398be706296738"
    )
    assert codec["trainable_during_B3"] is False
    assert codec["normalization_refit_allowed"] is False


def test_phase3_b3_noise_is_global_shared_and_present_in_every_block() -> None:
    record = _manifest()
    noise = record["functional_noise"]
    assert noise["raw_distribution"] == "standard_normal"
    assert noise["raw_dimension"] == 32
    assert noise["embedded_dimension"] == 256
    assert noise["spatial_semantics"] == "one_global_vector_shared_across_all_tokens"
    assert noise["layer_semantics"] == "same_embedding_supplied_to_all_16_blocks"
    assert noise["member_semantics"] == "independent_raw_vector_per_ensemble_member"
    assert noise["noise_layers"] == "all"
    assert noise["adapter_last_weight_post_init_multiplier"] == 0.01
    assert noise["adapter_last_bias_zero"] is True
    assert record["model"]["transformer_blocks"] == 16


def test_phase3_b3_fair_crps_and_staged_schedule_are_frozen() -> None:
    record = _manifest()
    objective = record["objective"]
    assert objective["name"] == "equal_channel_decoded_standardized_field_fair_CRPS"
    assert objective["ensemble_members_during_training"] == 2
    assert objective["codec_decode_in_gradient_path"] is True
    assert objective["codec_parameters_trainable"] is False
    assert objective["physics_derived_loss_allowed"] is False

    training = record["prospective_full_training"]
    assert training["authorized"] is False
    assert training["seed"] == 1701
    assert training["epochs"] == 100
    assert training["targets_per_epoch"] == 430
    assert training["optimizer_steps_per_epoch"] == 27
    assert training["total_optimizer_steps"] == 2700
    assert training["common_parameter_peak_learning_rate"] == 3.0e-5
    assert training["new_parameter_peak_learning_rate"] == 1.0e-4
    assert training["warmup_optimizer_steps"] == 270
    assert training["early_stopping"] is False


def test_phase3_b3_validation_bank_and_interface_are_explicit() -> None:
    record = _manifest()
    validation = record["validation"]
    assert validation["target_count"] == 126
    assert validation["ensemble_members"] == 2
    assert validation["noise_bank_generator"] == (
        "numpy.random.Generator(numpy.random.PCG64(31003))"
    )
    assert validation["noise_bank_shape"] == [126, 2, 32]
    assert validation["physics_metrics_select_checkpoint"] is False

    interface = record["forecast_interface"]
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


def test_phase3_b3_upstream_and_unchanged_local_sources_are_locked() -> None:
    record = _manifest()
    upstream = record["provenance"]["official_repository"]
    assert upstream["url"] == "https://github.com/cddcam/lola_crps"
    assert upstream["commit"] == "7643376c2949717ee5c2c840584689f529ba77a5"
    assert upstream["license"] == "MIT"
    assert upstream["source_files"]["lola/nn/vit.py"] == (
        "8d1607a4ab4c851e8258e6e5556f56d667362449c65b6db081c2f052f1eb7d34"
    )
    assert upstream["source_files"]["lola/crps.py"] == (
        "24ea94d9cb33b76a93fa12079d383fc7448123e3c1cdf7a2f91c9dd23e9eca12"
    )

    for relative, expected in record["provenance"][
        "existing_Paper0_sources_locked_against_initial_B3_modification"
    ].items():
        assert _sha256(ROOT / relative) == expected
