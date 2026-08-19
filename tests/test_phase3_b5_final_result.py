"""Regression locks for the complete B5 one-seed localization package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "paper0/results/phase3_b5_residual_edm_one_seed_localization_6901661.json"
)
FIGURES = ROOT / "paper0/figures/phase3_b5"
SUMMARIZER = ROOT / "paper0/tools/summarize_b5_residual_edm_gate.py"
PLOTTER = ROOT / "paper0/tools/plot_b5_residual_edm_gate.py"
EXPECTED_RESULT_SHA256 = (
    "ae10349b98394914f6a87dc99bebdc965056a941356f32b0392e261169cbf1f6"
)


def load() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_B5_localization_result_is_byte_locked_and_85604_only() -> None:
    assert digest(RESULT) == EXPECTED_RESULT_SHA256
    result = load()
    assert result["scope"] == (
        "phase3_B5_joint_residual_EDM_one_seed_localization_85604"
    )
    assert result["status"] == "completed_reproducible_failure_localization"
    assert result["development_run"] == "85604"
    assert result["held_out_85606_read"] is False
    assert result["guard_frames_read"] is False
    assert result["training_performed_by_this_tool"] is False
    assert result["inference_performed_by_this_tool"] is False
    assert result["truth_scoring_performed_by_this_tool"] is False
    assert result["thresholds_changed_by_this_tool"] is False


def test_B5_authoritative_input_hashes_are_frozen() -> None:
    assert load()["input_artifacts"] == {
        "b3_result": {
            "sha256": "f8ac75e65586aaa40b905ad4d447f15cad218deaa1119246d518b7730ede0dd3"
        },
        "b4_result": {
            "sha256": "4c07a7f4886c14ca2e53d6e322fe309e5efde1f76ab2ed779a3acd14d110f6be"
        },
        "b4_score": {
            "sha256": "055d81979f46a96bc0c983e0ef2f387f3032a2505117849089047e4f00b67dd3"
        },
        "evaluation_result": {
            "sha256": "bb8b5edfb8e01c45cdc8002895d4479d92728671cb023fdea106e9136bfca0f6"
        },
        "final_gate": {
            "sha256": "a1d9cf00de0a2b0b3cc0c13d31c727420214040dcbf575afa67c6ae64015974b"
        },
        "h1_comparator": {
            "sha256": "2b04c10971e6d38ee439e33aa0b5331305acf16b38a96e7952fb26046049b5d2"
        },
        "history": {
            "sha256": "8236c79dd2e83d7bea0b8f7afaf2162306d911195bd46f11834e482a6d7b7401"
        },
        "score": {
            "sha256": "c81c0e06313c652816be77025c2b42bbfce10728df7ac14787e00edf7d978ba6"
        },
        "training_result": {
            "sha256": "31d17363261cbe3e007bdeb0862760cfb731b2490da7d76b37f474f8fdb93068"
        },
    }


def test_B5_task_and_training_selection_are_exact() -> None:
    result = load()
    assert result["task"] == {
        "DCAE_or_latent_representation_used_for_residual": False,
        "absolute_time_input_used": False,
        "context_frames": 1,
        "fields": ["Ne", "Pe", "Pi", "phi", "Vi"],
        "future_frames": 1,
        "horizon_microseconds": 3.131905426352636,
        "mode_mapping": "n=5k",
        "model": "frozen_H1_mean_plus_joint_field_space_EDM_residual",
        "physics_derived_training_loss_used": False,
        "zperiod": 5,
    }
    training = result["training"]
    assert training["job_id"] == "6901531"
    assert training["selected_completed_epoch"] == 100
    assert training["selected_optimizer_step"] == 10_800
    assert training["selected_validation_EDM_loss"] == 0.30807498889783075
    assert training["selected_checkpoint_sha256"] == (
        "255904ef362c4d3f0fdb873131cd0b30bc02ea384e76e244d50698bd50df0c72"
    )
    assert training["checkpoint_reload_bitwise_exact"] is True
    assert len(training["epochs"]) == 100
    assert sum(
        item["validation_EDM_loss"] is not None for item in training["epochs"]
    ) == 20


def test_B5_marginal_gain_does_not_hide_mode_and_joint_failure() -> None:
    result = load()
    aggregate = result["field_and_marginal"]["aggregate"]
    assert aggregate["ensemble_mean_mae"] == 0.04373793076402049
    assert aggregate["ensemble_mean_rmse"] == 0.07490702592692233
    assert aggregate["fair_crps"] == 0.0314454060512613
    assert aggregate["spread_skill"] == 0.801695328375038
    assert aggregate["all_fields_nonzero_spread"] is True
    assert result["field_and_marginal"][
        "fields_passing_primary_spread_skill"
    ] == 1
    assert result["field_and_marginal"][
        "private_flux_fields_passing_I31"
    ] == 3

    counts = result["spectral_and_cross_field"]["counts"]
    assert counts["member_expected_power_ratio"] == {"passing": 13, "total": 15}
    assert counts["ensemble_mean_realization_coherence"] == {
        "passing": 4,
        "total": 15,
    }
    assert counts["mode_power_spread_skill"] == {"passing": 0, "total": 15}
    assert counts["mode_power_I31_coverage"] == {"passing": 0, "total": 15}
    assert counts["cross_phase"] == {"passing": 9, "total": 9}
    assert counts["cross_coherence_change"] == {"passing": 6, "total": 9}
    assert counts["cross_projection_spread_skill"] == {
        "passing": 0,
        "total": 18,
    }
    assert counts["cross_projection_I31_coverage"] == {
        "passing": 0,
        "total": 18,
    }


def test_B5_transport_and_chronological_failures_are_explicit() -> None:
    result = load()
    transport = result["transport"]
    assert transport["memberwise_nonlinear_operator"] is True
    assert transport["transport_of_ensemble_mean_fields_used"] is False
    assert transport["separatrix_calibrated_count"] == 0
    assert transport["separatrix_fair_crps_better_than_H1_count"] == 4
    assert transport["quantities"]["particle"]["strict_face_contributions"][
        "relative_l2"
    ] == 0.7050493016132817
    assert transport["quantities"]["particle"]["separatrix_wedge"][
        "correlation"
    ] == 0.7927948269942968
    assert transport["quantities"]["particle"]["separatrix_wedge"][
        "spread_skill"
    ] == 0.4851085770253902

    chronology = result["chronology"]
    assert chronology["joint_blocks_passing"] == 0
    assert chronology["joint_blocks_required"] == 5
    assert len(chronology["blocks"]) == 6
    assert all(not block["all_families_pass"] for block in chronology["blocks"])
    assert chronology["blocks"][4]["target_frames"] == [582, 603]
    assert chronology["blocks"][4]["B5_rmse"] == 0.08591386075260214


def test_B5_gate_stops_O3_assimilation_ranking_and_85606() -> None:
    gate = load()["gate"]
    assert gate["status"] == "completed_failed_frozen_B5_one_seed_gate"
    assert gate["job_id"] == "6901661"
    assert gate["passes_complete_one_seed_gate"] is False
    assert gate["integrity"] == {
        "passes": True,
        "check_count": 120,
        "failed_check_count": 0,
    }
    assert gate["families"] == {
        "field": {
            "blocks_passing": 0,
            "blocks_required": 5,
            "check_count": 54,
            "failed_check_count": 4,
            "passes": False,
        },
        "spectral": {
            "blocks_passing": 0,
            "blocks_required": 5,
            "check_count": 148,
            "failed_check_count": 83,
            "passes": False,
        },
        "transport": {
            "blocks_passing": 0,
            "blocks_required": 5,
            "check_count": 77,
            "failed_check_count": 7,
            "passes": False,
        },
    }
    assert gate["disposition"] == "B5_one_step_gate_failed_localize_without_retuning"
    for key in (
        "O3_protocol_may_be_written",
        "O3_launch_allowed",
        "additional_seed_training_authorized",
        "held_out_85606_access_allowed",
        "assimilation_allowed",
        "diagnostic_ranking_allowed",
    ):
        assert gate[key] is False


def test_all_six_B5_figure_pairs_exist_and_define_their_metrics() -> None:
    stems = (
        "b5-training-curves",
        "b5-model-comparison",
        "b5-field-spectral-localization",
        "b5-cross-field-localization",
        "b5-transport-localization",
        "b5-chronological-localization",
    )
    assert {path.name for path in FIGURES.iterdir()} == {
        f"{stem}.{suffix}" for stem in stems for suffix in ("svg", "png")
    }
    for stem in stems:
        assert (FIGURES / f"{stem}.svg").stat().st_size > 20_000
        assert (FIGURES / f"{stem}.png").stat().st_size > 50_000

    labels = "\n".join(
        (FIGURES / f"{stem}.svg").read_text(encoding="utf-8") for stem in stems
    )
    for phrase in (
        "fixed-seed validation",
        "Fair CRPS / deterministic H1 MAE",
        "n=5k",
        "Member-expected cross-phase error",
        "Strict facewise relative L2 error",
        "chronological block",
    ):
        assert phrase in labels


def test_B5_localization_tools_are_bound_to_locked_inputs() -> None:
    summarizer = SUMMARIZER.read_text(encoding="utf-8")
    plotter = PLOTTER.read_text(encoding="utf-8")
    assert "EXPECTED_SHA256" in summarizer
    assert "held-out 85606 path is prohibited" in summarizer
    assert "training_performed_by_this_tool" in summarizer
    assert "truth_scoring_performed_by_this_tool" in summarizer
    assert "phase3_B5_joint_residual_EDM_one_seed_localization_85604" in plotter
    assert "PASS/FAIL is not inferred from color" in plotter
