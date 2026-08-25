"""Fail-closed authority checks for the frozen persistent-pilot evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .codec_training import sha256_path
from .model_data import assert_development_path, load_strict_json
from .persistent_global_local_forecast import (
    PGL_EVALUATION_BLOCKS,
    PGL_EVALUATION_STARTS,
    PGL_SCIENTIFIC_SEED_BANK_SHA256,
)


PGL_EVALUATION_SCOPE = (
    "post_ecrd_old_85604_persistent_global_local_physics_evaluation"
)
PGL_EVALUATION_STATUS = "frozen_after_seed1702_state_gate_before_forecast"
PGL_TRAINING_RESULT_SHA256 = (
    "0f3b9e71d32b16269ec93e1601af1d569827b7d67ed20884c38fe7015abe10b6"
)
PGL_SELECTED_CHECKPOINT_SHA256 = (
    "4430eb1af96ee48faac80420227be42db363f5703712726b86d02836d42937eb"
)


def resolve_locked_path(record: Mapping[str, Any], *, root: Path, label: str) -> Path:
    value = Path(str(record.get("path", "")))
    path = value if value.is_absolute() else root / value
    assert_development_path(path)
    digest = str(record.get("sha256", ""))
    if len(digest) != 64 or not path.is_file() or sha256_path(path) != digest:
        raise ValueError(f"{label} path or SHA-256 differs")
    return path.resolve(strict=True)


def authorize_pgl_evaluation_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    paper0_root: Path,
) -> dict[str, Path]:
    """Validate the post-training, pre-forecast evaluation contract."""

    if sha256_path(manifest_path) != str(manifest_sha256):
        raise ValueError("persistent evaluation manifest SHA-256 differs")
    expected_flags = {
        "schema_version": 1,
        "scope": PGL_EVALUATION_SCOPE,
        "status": PGL_EVALUATION_STATUS,
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "new_nersc_data_read": False,
        "new_nersc_data_access_allowed": False,
        "guard_frames_read_allowed": False,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "physics_derived_loss_used": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "steering_allowed": False,
        "confirmation_seed_training_allowed": False,
        "wandb_required": True,
        "zperiod": 5,
        "mode_mapping": "n=5k",
    }
    if any(manifest.get(name) != value for name, value in expected_flags.items()):
        raise ValueError("persistent evaluation scope flags differ")
    population = manifest.get("forecast_population", {})
    expected_population = {
        "current_frame_blocks": {
            name: list(values) for name, values in PGL_EVALUATION_BLOCKS.items()
        },
        "current_frames": list(PGL_EVALUATION_STARTS),
        "start_count": 36,
        "ensemble_members": 32,
        "future_frames": 4,
        "primary_horizons": [1, 4],
        "fields": ["Ne", "Pe", "Pi", "phi", "Vi"],
        "forecast_axes": [
            "start",
            "ensemble_member",
            "future_time",
            "channel",
            "x",
            "y",
            "stored_toroidal_z",
        ],
        "target_truth_read_during_generation": False,
    }
    if population != expected_population:
        raise ValueError("persistent evaluation population differs")
    sampler = manifest.get("sampler", {})
    expected_sampler = {
        "seed_bank_sha256": PGL_SCIENTIFIC_SEED_BANK_SHA256,
        "seed_row": "one_frame_target_t_plus_1_minus_498",
        "initial_noise": "persistent_global_local_structured_noise",
        "steps": 18,
        "network_evaluations_per_member": 35,
        "member_batch_size": 8,
        "posthoc_spread_multiplier": False,
        "member_interaction": False,
    }
    if sampler != expected_sampler:
        raise ValueError("persistent evaluation sampler differs")
    expected_gates = {
        "field": {
            "fair_CRPS_strictly_below_selected_mean_MAE_at_horizons": [1, 4],
            "maximum_per_field_corrected_spread_skill": 1.5,
        },
        "spectral": {
            "bands": ["k1_3", "k4_5", "k6_7"],
            "maximum_candidate_over_parent_median_abs_log_power_error": 1.1,
            "primary_horizon": 4,
        },
        "cross_field": {
            "pair": "Ne-phi",
            "stored_k": [1, 7],
            "complex_cross_spectrum_error_strictly_improves": True,
            "maximum_phase_error_increase_degrees": 2.0,
            "primary_horizon": 4,
        },
        "spatial_transport_covariance": {
            "maximum_median_relative_frobenius_error": 0.9,
            "primary_horizon": 4,
        },
        "local_transport": {
            "calibrated_interval": [0.8, 1.25],
            "minimum_quantities_in_interval": 3,
            "maximum_any_ratio": 1.4,
            "primary_horizon": 4,
        },
        "integrated_transport": {
            "minimum_median_corrected_spread_skill": 0.6,
            "maximum_candidate_over_parent_median_relative_L2": 1.05,
            "primary_horizon": 4,
        },
        "all_families_required": True,
    }
    if manifest.get("physics_gates") != expected_gates:
        raise ValueError("persistent evaluation physics gates differ")
    bootstrap = manifest.get("bootstrap", {})
    if bootstrap != {
        "method": "paired_non_circular_selected_start_block_bootstrap",
        "block_length_selected_starts": 3,
        "replicates": 2000,
        "seed": 85604405,
        "resample_each_chronological_block_separately": True,
        "conditional_on_single_85604_run": True,
        "used_as_pass_fail_gate": False,
    }:
        raise ValueError("persistent evaluation bootstrap differs")

    protocol = resolve_locked_path(
        manifest.get("protocol", {}),
        root=paper0_root,
        label="persistent physics protocol",
    )
    locks = manifest.get("evidence_locks", {})
    required = (
        "training_manifest",
        "training_result",
        "selected_checkpoint",
        "residual_scales",
        "scientific_seed_bank",
        "native_truth_result",
        "geometry_manifest",
        "geometry",
        "event_threshold_result",
    )
    if any(name not in locks for name in required):
        raise ValueError("persistent evaluation evidence lock is absent")
    paths = {
        name: resolve_locked_path(locks[name], root=paper0_root, label=name)
        for name in required
    }
    if (
        locks["training_result"].get("sha256") != PGL_TRAINING_RESULT_SHA256
        or locks["selected_checkpoint"].get("sha256")
        != PGL_SELECTED_CHECKPOINT_SHA256
        or locks["scientific_seed_bank"].get("sha256")
        != PGL_SCIENTIFIC_SEED_BANK_SHA256
    ):
        raise ValueError("persistent evaluation primary evidence identity differs")
    model = locks.get("model_dataset", {})
    root = Path(str(model.get("root", "")))
    assert_development_path(root)
    expected_model_hashes = {
        "manifest_sha256": "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18",
        "normalization_sha256": "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7",
        "artifact_index_sha256": "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01",
    }
    if any(model.get(name) != value for name, value in expected_model_hashes.items()):
        raise ValueError("persistent model-dataset lock differs")
    for filename, key in (
        ("model_dataset_manifest.json", "manifest_sha256"),
        ("normalization.json", "normalization_sha256"),
        ("artifact_sha256.txt", "artifact_index_sha256"),
    ):
        path = root / filename
        if not path.is_file() or sha256_path(path) != model[key]:
            raise ValueError(f"persistent model-dataset {filename} differs")
    paths["model_dataset"] = root.resolve(strict=True)
    paths["protocol"] = protocol

    code_locks = manifest.get("evaluation_code", {})
    if not code_locks:
        raise ValueError("persistent evaluation code locks are absent")
    for relative, record in code_locks.items():
        path = (paper0_root / relative).resolve(strict=True)
        if record != {"sha256": sha256_path(path)}:
            raise ValueError(f"persistent evaluation code lock {relative!r} differs")
    return paths


def load_authorized_training_result(path: Path) -> dict[str, Any]:
    result = load_strict_json(path)
    if (
        result.get("scope")
        != "post_ecrd_old_85604_persistent_global_local_pilot"
        or result.get("status") != "completed"
        or result.get("development_run") != "85604"
        or result.get("seed") != 1702
        or result.get("completed_epochs") != 20
        or result.get("completed_optimizer_steps") != 4280
        or result.get("mechanical_gate", {}).get("passed") is not True
        or result.get("state_gate", {}).get("passed") is not True
        or result.get("physics_evaluation_authorized") is not True
        or result.get("physics_diagnostics_scored") is not False
        or result.get("physics_derived_loss_used") is not False
        or result.get("held_out_85606_read") is not False
        or result.get("new_nersc_data_read") is not False
        or result.get("guard_frames_read") is not False
        or result.get("selected_checkpoint", {}).get("sha256")
        != PGL_SELECTED_CHECKPOINT_SHA256
        or result.get("selected_checkpoint", {}).get("completed_epoch") != 20
    ):
        raise ValueError("persistent training result does not authorize evaluation")
    return result

