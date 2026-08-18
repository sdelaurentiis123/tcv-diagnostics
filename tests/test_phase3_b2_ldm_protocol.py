"""Regression locks for the prospective Phase 3 B2 latent-diffusion protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper0/protocol/PHASE3_B2_LDM_PROTOCOL.md"
MANIFEST = ROOT / "paper0/manifests/phase3_b2_ldm_85604.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_phase3_b2_protocol_and_manifest_are_hash_locked() -> None:
    record = _manifest()
    assert _sha256(PROTOCOL) == (
        "22b20a1050b68aaada21efc3840b322da16bfa0ab4a870e03a5bfffc97ecfe44"
    )
    assert _sha256(MANIFEST) == (
        "75d6f812fc7d203d5c92c588515f4a59d4b3b0cb8688fdd7514b6d09b842f9c3"
    )
    assert record["protocol"] == {
        "path": "paper0/protocol/PHASE3_B2_LDM_PROTOCOL.md",
        "sha256": _sha256(PROTOCOL),
    }


def test_phase3_b2_scope_is_smoke_only_and_keeps_85606_closed() -> None:
    record = _manifest()
    assert record["development_run"] == "85604"
    assert record["sequestered_run"] == "85606"
    assert record["held_out_85606_access_allowed"] is False
    assert record["full_training_authorized"] is False
    assert record["authorized_scope"] == [
        "B2_LDM_H2_implementation",
        "B2_LDM_H2_CPU_tests",
        "B2_LDM_H2_single_seed_bounded_GPU_smoke_85604",
    ]
    forbidden = set(record["forbidden_scope"])
    assert "B2_full_training_before_new_authorization" in forbidden
    assert "B2_scientific_evaluation_before_frozen_evaluation_protocol" in forbidden
    assert "O3_or_longer_rollout" in forbidden
    assert "assimilation" in forbidden
    assert "diagnostic_ranking" in forbidden
    assert "85606_access" in forbidden


def test_phase3_b2_reuses_the_accepted_c5p_data_and_codec() -> None:
    record = _manifest()
    data = record["data"]
    assert data["fields"] == ["Ne", "Pe", "Pi", "phi", "Vi"]
    assert data["training_frames"] == [0, 432]
    assert data["training_targets"] == [2, 432]
    assert data["guard_frames"] == [432, 496]
    assert data["validation_frames"] == [496, 624]
    assert data["validation_targets"] == [498, 624]
    assert data["absolute_time_input_allowed"] is False
    assert data["future_truth_input_allowed"] is False
    assert data["zperiod"] == 5
    assert data["mode_mapping"] == "n=5k"
    assert data["cadence_microseconds"] == 3.131905426352636

    codec = record["codec"]
    assert codec["name"] == "dcae_l10"
    assert codec["family"] == "c5p"
    assert codec["latent_channels"] == 32
    assert codec["latent_grid"] == [16, 8, 22]
    assert codec["tokens_per_frame"] == 704
    assert codec["evaluation_mode_required"] is True
    assert codec["trainable_during_B2"] is False
    checkpoints = codec["selected_checkpoints"]
    assert [item["seed"] for item in checkpoints] == [1701, 1702, 1703]
    assert all(len(item["sha256"]) == 64 for item in checkpoints)


def test_phase3_b2_model_is_the_frozen_masked_h2_ldm() -> None:
    record = _manifest()
    model = record["model"]
    assert model["family"] == "LOLA_style_masked_latent_diffusion"
    assert model["representation"] == "C5P-dcae_l10"
    assert model["context_frames"] == 2
    assert model["future_frames"] == 1
    assert model["trajectory_frames"] == 3
    assert model["latent_channels"] == 32
    assert model["noise_time_features"] == 256
    assert model["hidden_channels"] == 512
    assert model["transformer_blocks"] == 16
    assert model["attention_heads"] == 4
    assert model["ffn_factor"] == 4
    assert model["condition_mask_channels"] == 1
    assert model["predictor_patch"] == [1, 2, 2, 1]
    assert model["tokens_per_trajectory"] == 2112
    assert model["qk_normalization"] is True
    assert model["rope"] is True
    assert model["global_attention"] is True
    assert model["physical_time_input"] is False
    assert model["shot_label_input"] is False
    assert model["physics_derived_loss_allowed"] is False

    noise = record["noise"]
    assert noise["schedule"] == "log_logit"
    assert noise["sigma_min"] == 0.001
    assert noise["sigma_max"] == 1000.0
    assert noise["scale"] == 1.0
    assert noise["shift"] == 0.0
    assert noise["EDM_noise_embedding_scale"] == 10.0
    assert noise["noise_time_distribution"] == "Uniform(0,1)"
    assert noise["loss_scope"] == "complete_three_frame_masked_trajectory"
    assert noise["training_loss"] == "LOLA_EDM_weighted_denoising_MSE"


def test_phase3_b2_budget_sampler_and_gate_are_prospective() -> None:
    record = _manifest()
    training = record["training"]
    assert training["seeds"] == [1701, 1702, 1703]
    assert training["epochs"] == 200
    assert training["targets_per_epoch"] == 430
    assert training["validation_targets"] == 126
    assert training["microbatch"] == 1
    assert training["gradient_accumulation"] == 16
    assert training["optimizer_steps_per_epoch"] == 27
    assert training["total_optimizer_steps"] == 5400
    assert training["optimizer"] == "AdamW"
    assert training["learning_rate"] == 0.0001
    assert training["betas"] == [0.9, 0.99]
    assert training["weight_decay"] == 0.0
    assert training["warmup_steps"] == 0
    assert training["scheduler"] == "cosine_to_zero_per_optimizer_step"
    assert training["gradient_clip"] == 1.0
    assert training["early_stopping"] is False
    assert training["validation_noise"].startswith("fixed_CPU_generator")

    sampler = record["sampler"]
    assert sampler == {
        "algorithm": "Azula_0.3.1_ABSampler",
        "independent_initial_noise_per_member": True,
        "order": 3,
        "scientific_ensemble_size": "not_yet_frozen",
        "start": 1.0,
        "steps": 16,
        "stop": 0.0,
    }

    smoke = record["implementation_gate"]["gpu_smoke"]
    assert smoke["accelerator"] == "one Rocky9 H100 or H200"
    assert smoke["seed"] == 1701
    assert smoke["training_targets_max"] == 16
    assert smoke["epochs_max"] == 2
    assert smoke["ensemble_members"] == 2
    assert smoke["wandb_online_required"] is True


def test_phase3_b2_provenance_rejects_the_legacy_checkpoint() -> None:
    record = _manifest()
    assert record["historical_control"]["checkpoint_use_allowed"] is False
    assert record["provenance"]["LOLA"]["source_commit"] == (
        "21a4354b327e6e5ee06da5075ba3bd1dd88c61f1"
    )
    assert record["provenance"]["LOLA"]["license"] == "MIT"
    assert record["provenance"]["azula"]["version"] == "0.3.1"
    assert record["provenance"]["azula"]["license"] == "MIT"
    assert record["wandb"]["smoke_and_full_mode"] == "online"
    assert record["wandb"]["successful_initialization_required_before_training"]


def test_phase3_b2_protocol_states_the_required_interpretation() -> None:
    text = " ".join(PROTOCOL.read_text().split())
    required = [
        "It does not consume a previous model prediction",
        "It tests that hypothesis",
        "not a Paper 0 B2 checkpoint",
        "Nonlinear diagnostics must be computed member by member",
        "Flux from the ensemble-mean fields is not an admissible replacement",
        "Full B2 training requires",
        "not scientific evidence",
    ]
    for phrase in required:
        assert phrase in text
