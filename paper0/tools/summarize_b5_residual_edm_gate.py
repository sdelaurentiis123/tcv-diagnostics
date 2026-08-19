#!/usr/bin/env python3
"""Build the compact, auditable B5 one-seed localization result.

This tool performs no training, inference, truth scoring, or threshold changes.
It verifies the immutable B5 artifacts and extracts the already-computed
training, one-step field, spectral, cross-field, transport, chronology, and
gate evidence into a reviewable result small enough to track in git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]

FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")
BANDS = ("k1_3", "k4_5", "k6_7")
CROSS_PAIRS = ("Ne-phi", "Pe-phi", "Pi-phi")
QUANTITIES = (
    "particle",
    "electron_internal_energy",
    "ion_internal_energy",
    "total_internal_energy",
)
MODEL_NAMES_FOR_COMPARISON = (
    "H1 deterministic",
    "B3 functional noise",
    "B4 PDE-Refiner",
    "B5 joint residual EDM",
)

EXPECTED_SHA256 = {
    "training_result": "31d17363261cbe3e007bdeb0862760cfb731b2490da7d76b37f474f8fdb93068",
    "evaluation_result": "bb8b5edfb8e01c45cdc8002895d4479d92728671cb023fdea106e9136bfca0f6",
    "history": "8236c79dd2e83d7bea0b8f7afaf2162306d911195bd46f11834e482a6d7b7401",
    "score": "c81c0e06313c652816be77025c2b42bbfce10728df7ac14787e00edf7d978ba6",
    "final_gate": "a1d9cf00de0a2b0b3cc0c13d31c727420214040dcbf575afa67c6ae64015974b",
    "b3_result": "f8ac75e65586aaa40b905ad4d447f15cad218deaa1119246d518b7730ede0dd3",
    "b4_result": "4c07a7f4886c14ca2e53d6e322fe309e5efde1f76ab2ed779a3acd14d110f6be",
    "b4_score": "055d81979f46a96bc0c983e0ef2f387f3032a2505117849089047e4f00b67dd3",
    "h1_comparator": "2b04c10971e6d38ee439e33aa0b5331305acf16b38a96e7952fb26046049b5d2",
}

EXPECTED_SCOPES = {
    "training_result": "B5_seed1701_full_training_and_data_only_selection_85604",
    "evaluation_result": "B5_residual_EDM_full_one_step_evaluation_85604",
    "score": "B5_residual_EDM_truth_separated_probabilistic_scoring_85604",
    "final_gate": "phase3_B5_residual_EDM_seed1701_scientific_gate_85604",
}

GATES = {
    "field_spread_skill": [0.80, 1.25],
    "private_flux_I31_coverage": [0.75, 0.995],
    "power_ratio": [0.75, 1.30],
    "realization_coherence_min": 0.80,
    "cross_phase_absolute_error_degrees_max": 20.0,
    "cross_coherence_absolute_change_max": 0.15,
    "mode_or_cross_projection_spread_skill": [0.67, 1.50],
    "mode_or_cross_projection_I31_coverage": [0.75, 0.995],
    "strict_transport_relative_l2_max": 0.40,
    "separatrix_relative_l2_max": 0.30,
    "separatrix_correlation_min": 0.80,
    "separatrix_spread_skill": [0.67, 1.50],
    "separatrix_I31_nominal": 31.0 / 33.0,
    "separatrix_I31_coverage_tolerance": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--evaluation-result", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--final-gate", type=Path, required=True)
    parser.add_argument(
        "--b3-result",
        type=Path,
        default=ROOT / "paper0/results/phase3_b3_fgn_one_seed_gate_6899224.json",
    )
    parser.add_argument(
        "--b4-result",
        type=Path,
        default=ROOT / "paper0/results/phase3_b4_pde_refiner_one_seed_gate_6901285.json",
    )
    parser.add_argument("--b4-score", type=Path, required=True)
    parser.add_argument("--h1-comparator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def load_history(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"history line {line_number} is not an object")
        records.append(value)
    return records


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def in_range(value: float, bounds: Iterable[float]) -> bool:
    lower, upper = (float(item) for item in bounds)
    return lower <= value <= upper


def verify_inputs(paths: Mapping[str, Path]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, path in paths.items():
        resolved = path.resolve(strict=True)
        if "85606" in str(resolved).lower():
            raise ValueError(f"held-out 85606 path is prohibited: {resolved}")
        digest = sha256_path(resolved)
        if digest != EXPECTED_SHA256[name]:
            raise ValueError(
                f"{name} SHA-256 differs: expected {EXPECTED_SHA256[name]}, got {digest}"
            )
        actual[name] = digest
    return actual


def _field_summary(score: Mapping[str, Any]) -> dict[str, Any]:
    field_root = score["field_and_marginal_calibration"]
    eligible = field_root["regions"]["eligible_union"]
    private_flux = field_root["regions"]["private_flux"]
    fields: dict[str, Any] = {}
    for field in FIELDS:
        value = eligible["fields"][field]
        private = private_flux["fields"][field]
        spread = finite_number(
            value["corrected_spread_skill"]["ratio"], f"{field} spread-skill"
        )
        private_i31 = finite_number(
            private["order_statistic_intervals"]["I31"]["empirical_coverage"],
            f"{field} private-flux I31 coverage",
        )
        fields[field] = {
            "ensemble_mean_mae": finite_number(
                value["ensemble_mean"]["mae"], f"{field} MAE"
            ),
            "ensemble_mean_rmse": finite_number(
                value["ensemble_mean"]["rmse"], f"{field} RMSE"
            ),
            "fair_crps": finite_number(value["fair_crps"], f"{field} fair CRPS"),
            "spread_skill": spread,
            "spread_skill_passes_primary": in_range(
                spread, GATES["field_spread_skill"]
            ),
            "I31_coverage": finite_number(
                value["order_statistic_intervals"]["I31"]["empirical_coverage"],
                f"{field} I31 coverage",
            ),
            "private_flux_spread_skill": finite_number(
                private["corrected_spread_skill"]["ratio"],
                f"{field} private-flux spread-skill",
            ),
            "private_flux_I31_coverage": private_i31,
            "private_flux_I31_passes": in_range(
                private_i31, GATES["private_flux_I31_coverage"]
            ),
        }
    aggregate = eligible["aggregate"]
    return {
        "aggregate": {
            "ensemble_mean_mae": finite_number(
                aggregate["equal_channel_ensemble_mean_mae"], "aggregate MAE"
            ),
            "ensemble_mean_rmse": finite_number(
                aggregate["equal_channel_ensemble_mean_rmse"], "aggregate RMSE"
            ),
            "fair_crps": finite_number(
                aggregate["equal_channel_fair_crps"], "aggregate fair CRPS"
            ),
            "corrected_rms_spread": finite_number(
                aggregate["equal_channel_corrected_rms_spread"],
                "aggregate corrected RMS spread",
            ),
            "spread_skill": finite_number(
                aggregate["equal_channel_corrected_spread_skill_ratio"],
                "aggregate spread-skill",
            ),
            "all_fields_nonzero_spread": bool(aggregate["all_fields_nonzero_spread"]),
        },
        "fields": fields,
        "fields_passing_primary_spread_skill": sum(
            item["spread_skill_passes_primary"] for item in fields.values()
        ),
        "private_flux_fields_passing_I31": sum(
            item["private_flux_I31_passes"] for item in fields.values()
        ),
        "M16_vs_M32": field_root["M16_vs_M32_stability"],
    }


def _spectral_summary(score: Mapping[str, Any]) -> dict[str, Any]:
    root = score["spectral_and_cross_field"]["overall"]
    field_bands: dict[str, Any] = {}
    power_passes = 0
    coherence_passes = 0
    spread_passes = 0
    coverage_passes = 0
    for field in FIELDS:
        field_bands[field] = {}
        for band in BANDS:
            source = root["toroidal_field_power"][field]["bands"][band]
            calibration = source["per_target_band_power_calibration"]
            power = finite_number(
                source["member_expected_power_ratio"], f"{field} {band} power ratio"
            )
            coherence = finite_number(
                source["ensemble_mean_realization_coherence_with_truth"],
                f"{field} {band} realization coherence",
            )
            spread = finite_number(
                calibration["primary_M32"]["corrected_spread_skill"][
                    "spread_skill_ratio"
                ],
                f"{field} {band} mode spread-skill",
            )
            coverage = finite_number(
                calibration["order_statistic_intervals"]["I31"][
                    "empirical_coverage"
                ],
                f"{field} {band} mode I31 coverage",
            )
            record = {
                "stored_k": source["stored_k"],
                "full_torus_n": source["full_torus_n"],
                "member_expected_power_ratio": power,
                "power_ratio_passes": in_range(power, GATES["power_ratio"]),
                "ensemble_mean_realization_coherence": coherence,
                "realization_coherence_passes": (
                    coherence >= GATES["realization_coherence_min"]
                ),
                "mode_power_spread_skill": spread,
                "mode_power_spread_skill_passes": in_range(
                    spread, GATES["mode_or_cross_projection_spread_skill"]
                ),
                "mode_power_I31_coverage": coverage,
                "mode_power_I31_coverage_passes": in_range(
                    coverage, GATES["mode_or_cross_projection_I31_coverage"]
                ),
            }
            power_passes += int(record["power_ratio_passes"])
            coherence_passes += int(record["realization_coherence_passes"])
            spread_passes += int(record["mode_power_spread_skill_passes"])
            coverage_passes += int(record["mode_power_I31_coverage_passes"])
            field_bands[field][band] = record

    cross_bands: dict[str, Any] = {}
    phase_passes = 0
    cross_coherence_passes = 0
    projection_spread_passes = 0
    projection_coverage_passes = 0
    for pair in CROSS_PAIRS:
        cross_bands[pair] = {}
        for band in BANDS:
            source = root["toroidal_cross_field"][pair]["bands"][band]
            signed_phase = finite_number(
                source["summed_complex_cross_phase_error_degrees"],
                f"{pair} {band} cross-phase error",
            )
            coherence_change = finite_number(
                source["truth_amplitude_weighted_absolute_coherence_change"],
                f"{pair} {band} cross-coherence change",
            )
            projections: dict[str, Any] = {}
            for projection in ("real", "imaginary"):
                calibration = source["per_target_cross_projection_calibration"][
                    projection
                ]
                spread = finite_number(
                    calibration["primary_M32"]["corrected_spread_skill"][
                        "spread_skill_ratio"
                    ],
                    f"{pair} {band} {projection} spread-skill",
                )
                coverage = finite_number(
                    calibration["order_statistic_intervals"]["I31"][
                        "empirical_coverage"
                    ],
                    f"{pair} {band} {projection} I31 coverage",
                )
                spread_ok = in_range(
                    spread, GATES["mode_or_cross_projection_spread_skill"]
                )
                coverage_ok = in_range(
                    coverage, GATES["mode_or_cross_projection_I31_coverage"]
                )
                projection_spread_passes += int(spread_ok)
                projection_coverage_passes += int(coverage_ok)
                projections[projection] = {
                    "spread_skill": spread,
                    "spread_skill_passes": spread_ok,
                    "I31_coverage": coverage,
                    "I31_coverage_passes": coverage_ok,
                }
            phase_ok = (
                abs(signed_phase)
                <= GATES["cross_phase_absolute_error_degrees_max"]
            )
            coherence_ok = (
                coherence_change <= GATES["cross_coherence_absolute_change_max"]
            )
            phase_passes += int(phase_ok)
            cross_coherence_passes += int(coherence_ok)
            cross_bands[pair][band] = {
                "signed_cross_phase_error_degrees": signed_phase,
                "absolute_cross_phase_error_degrees": abs(signed_phase),
                "cross_phase_passes": phase_ok,
                "absolute_cross_coherence_change": coherence_change,
                "cross_coherence_change_passes": coherence_ok,
                "projections": projections,
            }
    return {
        "mode_mapping": {"zperiod": 5, "rule": "n=5k"},
        "field_bands": field_bands,
        "cross_field_bands": cross_bands,
        "counts": {
            "member_expected_power_ratio": {"passing": power_passes, "total": 15},
            "ensemble_mean_realization_coherence": {
                "passing": coherence_passes,
                "total": 15,
            },
            "mode_power_spread_skill": {"passing": spread_passes, "total": 15},
            "mode_power_I31_coverage": {"passing": coverage_passes, "total": 15},
            "cross_phase": {"passing": phase_passes, "total": 9},
            "cross_coherence_change": {
                "passing": cross_coherence_passes,
                "total": 9,
            },
            "cross_projection_spread_skill": {
                "passing": projection_spread_passes,
                "total": 18,
            },
            "cross_projection_I31_coverage": {
                "passing": projection_coverage_passes,
                "total": 18,
            },
        },
    }


def _transport_summary(
    score: Mapping[str, Any], h1: Mapping[str, Any]
) -> dict[str, Any]:
    source_quantities = score["memberwise_transport"]["overall"]["quantities"]
    comparator_quantities = h1["transport"]["overall"]["quantities"]
    quantities: dict[str, Any] = {}
    calibrated_count = 0
    fair_crps_better_count = 0
    for quantity in QUANTITIES:
        source = source_quantities[quantity]
        h1_source = comparator_quantities[quantity]
        reductions: dict[str, Any] = {}
        for name in ("strict_face_contributions", "separatrix_wedge"):
            reduction = source["reductions"][name]
            paired = reduction["ensemble_expected_paired_metrics"]
            probability = reduction["ensemble_probabilistic_metrics"]
            reductions[name] = {
                "relative_l2": finite_number(
                    paired["relative_l2"], f"{quantity} {name} relative L2"
                ),
                "correlation": finite_number(
                    paired["pearson_correlation"], f"{quantity} {name} correlation"
                ),
                "weighted_sign_disagreement": finite_number(
                    paired["weighted_sign_disagreement"],
                    f"{quantity} {name} sign disagreement",
                ),
                "normalized_bias": finite_number(
                    paired["normalized_bias"], f"{quantity} {name} bias"
                ),
                "fair_crps": finite_number(
                    probability["fair_crps"], f"{quantity} {name} fair CRPS"
                ),
                "spread_skill": finite_number(
                    probability["corrected_spread_skill"]["ratio"],
                    f"{quantity} {name} spread-skill",
                ),
                "I31_coverage": finite_number(
                    probability["order_statistic_intervals"]["I31"][
                        "empirical_coverage"
                    ],
                    f"{quantity} {name} I31 coverage",
                ),
            }
        separatrix = reductions["separatrix_wedge"]
        h1_absolute_error = finite_number(
            h1_source["separatrix_absolute_error"],
            f"{quantity} H1 separatrix absolute error",
        )
        separatrix["fair_crps_relative_to_H1_absolute_error"] = (
            separatrix["fair_crps"] / h1_absolute_error
        )
        spread_ok = in_range(
            separatrix["spread_skill"], GATES["separatrix_spread_skill"]
        )
        coverage_ok = (
            abs(
                separatrix["I31_coverage"]
                - GATES["separatrix_I31_nominal"]
            )
            <= GATES["separatrix_I31_coverage_tolerance"]
        )
        separatrix["spread_skill_passes"] = spread_ok
        separatrix["I31_coverage_passes"] = coverage_ok
        separatrix["probabilistically_calibrated"] = spread_ok and coverage_ok
        calibrated_count += int(separatrix["probabilistically_calibrated"])
        fair_crps_better_count += int(
            separatrix["fair_crps_relative_to_H1_absolute_error"] < 1.0
        )
        quantities[quantity] = {
            "H1_strict_relative_l2": finite_number(
                h1_source["strict_face_contributions"]["relative_l2"],
                f"{quantity} H1 strict relative L2",
            ),
            "H1_separatrix_relative_l2": finite_number(
                h1_source["separatrix_wedge"]["relative_l2"],
                f"{quantity} H1 separatrix relative L2",
            ),
            "strict_face_contributions": reductions["strict_face_contributions"],
            "separatrix_wedge": separatrix,
        }
    return {
        "memberwise_nonlinear_operator": bool(
            score["memberwise_transport"]["overall"][
                "nonlinear_operator_applied_per_member_before_reduction"
            ]
        ),
        "transport_of_ensemble_mean_fields_used": bool(
            score["memberwise_transport"]["overall"][
                "transport_of_ensemble_mean_fields_used"
            ]
        ),
        "quantities": quantities,
        "separatrix_calibrated_count": calibrated_count,
        "separatrix_fair_crps_better_than_H1_count": fair_crps_better_count,
    }


def _chronology_summary(
    score: Mapping[str, Any],
    b4_score: Mapping[str, Any],
    h1: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    b5_blocks = score["field_and_marginal_calibration"][
        "chronological_blocks_eligible_union"
    ]
    b4_blocks = b4_score["field_and_marginal_calibration"][
        "chronological_blocks_eligible_union"
    ]
    h1_blocks = h1["field"]["chronological_blocks"]
    if not (len(b5_blocks) == len(b4_blocks) == len(h1_blocks) == 6):
        raise ValueError("expected six matching chronological validation blocks")
    result_blocks: list[dict[str, Any]] = []
    for index, (b5, b4, h1_block) in enumerate(zip(b5_blocks, b4_blocks, h1_blocks)):
        frames = list(b5["target_frames"])
        if frames != list(b4["target_frames"]) or frames != list(
            h1_block["target_frames"]
        ):
            raise ValueError(f"chronological block {index} frame ranges differ")
        failures: dict[str, Any] = {}
        for family in ("field", "spectral", "transport"):
            gate_block = gate["acceptance"]["families"][family][
                "chronological_blocks"
            ][index]
            failed = int(gate_block["failed_check_count"])
            total = int(gate_block["check_count"])
            failures[family] = {
                "failed": failed,
                "total": total,
                "fraction_failed": failed / total,
            }
        aggregate = b5["aggregate"]
        result_blocks.append(
            {
                "index": index,
                "target_frames": frames,
                "midpoint_frame": (frames[0] + frames[1] - 1) / 2.0,
                "H1_rmse": finite_number(
                    h1_block["aggregate_equal_channel_rmse_standardized"],
                    f"block {index} H1 RMSE",
                ),
                "H1_mae": finite_number(
                    h1_block["aggregate_equal_channel_mae_standardized"],
                    f"block {index} H1 MAE",
                ),
                "B4_rmse": finite_number(
                    b4["aggregate"]["equal_channel_ensemble_mean_rmse"],
                    f"block {index} B4 RMSE",
                ),
                "B4_fair_crps": finite_number(
                    b4["aggregate"]["equal_channel_fair_crps"],
                    f"block {index} B4 fair CRPS",
                ),
                "B5_rmse": finite_number(
                    aggregate["equal_channel_ensemble_mean_rmse"],
                    f"block {index} B5 RMSE",
                ),
                "B5_fair_crps": finite_number(
                    aggregate["equal_channel_fair_crps"],
                    f"block {index} B5 fair CRPS",
                ),
                "B5_spread_skill": finite_number(
                    aggregate["equal_channel_corrected_spread_skill_ratio"],
                    f"block {index} B5 spread-skill",
                ),
                "failed_check_counts": failures,
                "all_families_pass": all(
                    record["failed"] == 0 for record in failures.values()
                ),
            }
        )
    return {
        "blocks": result_blocks,
        "joint_blocks_passing": sum(block["all_families_pass"] for block in result_blocks),
        "joint_blocks_required": 5,
    }


def _training_summary(
    training: Mapping[str, Any], history: list[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(history) != 100:
        raise ValueError(f"expected 100 training-history records, got {len(history)}")
    epochs: list[dict[str, Any]] = []
    candidate_epochs: list[int] = []
    for expected_epoch, source in enumerate(history, 1):
        epoch = int(source["completed_epoch"])
        if epoch != expected_epoch:
            raise ValueError("training history is not complete and chronological")
        validation = source.get("validation")
        record = {
            "completed_epoch": epoch,
            "optimizer_step": int(source["global_optimizer_step"]),
            "train_EDM_loss": finite_number(
                source["train_mean_EDM_loss"], f"epoch {epoch} train EDM loss"
            ),
            "train_unweighted_MSE": finite_number(
                source["train_mean_unweighted_MSE"],
                f"epoch {epoch} train unweighted MSE",
            ),
            "mean_preclip_gradient_norm": finite_number(
                source["mean_preclip_gradient_norm"],
                f"epoch {epoch} mean gradient norm",
            ),
            "last_learning_rate": finite_number(
                source["last_learning_rate"], f"epoch {epoch} learning rate"
            ),
            "validation_EDM_loss": None,
            "validation_unweighted_MSE": None,
        }
        if validation is not None:
            candidate_epochs.append(epoch)
            record["validation_EDM_loss"] = finite_number(
                validation["mean_EDM_loss"], f"epoch {epoch} validation EDM loss"
            )
            record["validation_unweighted_MSE"] = finite_number(
                validation["mean_unweighted_MSE"],
                f"epoch {epoch} validation unweighted MSE",
            )
        epochs.append(record)
    if candidate_epochs != list(training["candidate_completed_epochs"]):
        raise ValueError("history candidate epochs differ from training result")
    selected_epoch = int(training["selected_completed_epoch"])
    selected = epochs[selected_epoch - 1]
    if selected["validation_EDM_loss"] != training["selected_validation"][
        "mean_EDM_loss"
    ]:
        raise ValueError("selected validation loss differs from history")
    return {
        "job_id": str(training["slurm_job_id"]),
        "status": training["status"],
        "model": training["model_config"],
        "optimizer_and_budget": training["config"],
        "parameter_count": int(training["parameter_count"]),
        "wall_seconds": finite_number(training["wall_seconds"], "training wall time"),
        "peak_cuda_GiB": finite_number(
            training["peak_cuda_GiB"], "training peak CUDA GiB"
        ),
        "selected_completed_epoch": selected_epoch,
        "selected_optimizer_step": int(training["selected_optimizer_step"]),
        "selected_validation_EDM_loss": selected["validation_EDM_loss"],
        "selected_checkpoint_sha256": training["artifacts"]["selected_checkpoint"][
            "sha256"
        ],
        "checkpoint_reload_bitwise_exact": bool(
            training["checkpoint_reload_bitwise_exact"]
        ),
        "epochs": epochs,
    }


def _model_comparison(
    b3: Mapping[str, Any],
    b4_score: Mapping[str, Any],
    b5_field: Mapping[str, Any],
    b5_spectral: Mapping[str, Any],
    b5_transport: Mapping[str, Any],
    h1: Mapping[str, Any],
) -> dict[str, Any]:
    h1_field = h1["field"]["overall"]
    h1_mae = finite_number(
        h1_field["aggregate_equal_channel_mae_standardized"], "H1 aggregate MAE"
    )
    h1_rmse = finite_number(
        h1_field["aggregate_equal_channel_rmse_standardized"], "H1 aggregate RMSE"
    )

    b4_field = _field_summary(b4_score)
    b4_spectral = _spectral_summary(b4_score)
    models = {
        "H1 deterministic": {
            "ensemble_mean_mae": h1_mae,
            "ensemble_mean_rmse": h1_rmse,
            "fair_crps": h1_mae,
            "spread_skill": 0.0,
            "power_checks_passing": None,
            "realization_coherence_checks_passing": None,
        },
        "B3 functional noise": {
            "ensemble_mean_mae": b3["field_and_marginal"]["aggregate"][
                "ensemble_mean_MAE"
            ],
            "ensemble_mean_rmse": b3["field_and_marginal"]["aggregate"][
                "ensemble_mean_RMSE"
            ],
            "fair_crps": b3["field_and_marginal"]["aggregate"]["fair_CRPS"],
            "spread_skill": b3["field_and_marginal"]["aggregate"][
                "corrected_spread_skill_ratio"
            ],
            "power_checks_passing": b3["spectral_and_cross_field"][
                "material_field_power_checks"
            ]["passing"],
            "realization_coherence_checks_passing": b3[
                "spectral_and_cross_field"
            ]["material_realization_coherence_checks"]["passing"],
        },
        "B4 PDE-Refiner": {
            "ensemble_mean_mae": b4_field["aggregate"]["ensemble_mean_mae"],
            "ensemble_mean_rmse": b4_field["aggregate"]["ensemble_mean_rmse"],
            "fair_crps": b4_field["aggregate"]["fair_crps"],
            "spread_skill": b4_field["aggregate"]["spread_skill"],
            "power_checks_passing": b4_spectral["counts"][
                "member_expected_power_ratio"
            ]["passing"],
            "realization_coherence_checks_passing": b4_spectral["counts"][
                "ensemble_mean_realization_coherence"
            ]["passing"],
        },
        "B5 joint residual EDM": {
            "ensemble_mean_mae": b5_field["aggregate"]["ensemble_mean_mae"],
            "ensemble_mean_rmse": b5_field["aggregate"]["ensemble_mean_rmse"],
            "fair_crps": b5_field["aggregate"]["fair_crps"],
            "spread_skill": b5_field["aggregate"]["spread_skill"],
            "power_checks_passing": b5_spectral["counts"][
                "member_expected_power_ratio"
            ]["passing"],
            "realization_coherence_checks_passing": b5_spectral["counts"][
                "ensemble_mean_realization_coherence"
            ]["passing"],
        },
    }
    for value in models.values():
        value["mae_relative_to_H1"] = value["ensemble_mean_mae"] / h1_mae
        value["rmse_relative_to_H1"] = value["ensemble_mean_rmse"] / h1_rmse
        value["fair_crps_relative_to_H1_MAE"] = value["fair_crps"] / h1_mae

    field_by_field: dict[str, Any] = {}
    for field in FIELDS:
        h1_field_mae = finite_number(
            h1_field["fields"][field]["mae"], f"H1 {field} MAE"
        )
        b3_field = b3["field_and_marginal"]["by_field"][field]
        field_by_field[field] = {
            "H1 deterministic": {
                "ensemble_mean_mae": h1_field_mae,
                "fair_crps": h1_field_mae,
                "spread_skill": None,
            },
            "B3 functional noise": {
                "ensemble_mean_mae": b3_field["ensemble_mean_MAE"],
                "fair_crps": b3_field["fair_CRPS"],
                "spread_skill": b3_field["spread_skill"],
            },
            "B4 PDE-Refiner": {
                "ensemble_mean_mae": b4_field["fields"][field][
                    "ensemble_mean_mae"
                ],
                "fair_crps": b4_field["fields"][field]["fair_crps"],
                "spread_skill": b4_field["fields"][field]["spread_skill"],
            },
            "B5 joint residual EDM": {
                "ensemble_mean_mae": b5_field["fields"][field][
                    "ensemble_mean_mae"
                ],
                "fair_crps": b5_field["fields"][field]["fair_crps"],
                "spread_skill": b5_field["fields"][field]["spread_skill"],
            },
        }
        for model in MODEL_NAMES_FOR_COMPARISON:
            field_by_field[field][model]["mae_relative_to_H1"] = (
                field_by_field[field][model]["ensemble_mean_mae"] / h1_field_mae
            )
            field_by_field[field][model]["fair_crps_relative_to_H1_MAE"] = (
                field_by_field[field][model]["fair_crps"] / h1_field_mae
            )

    strict_l2: dict[str, Any] = {}
    transport_by_quantity: dict[str, Any] = {}
    for quantity in QUANTITIES:
        h1_quantity = b5_transport["quantities"][quantity]
        b3_quantity = b3["transport"]["quantities"][quantity]
        b4_quantity = b4_score["memberwise_transport"]["overall"]["quantities"][
            quantity
        ]["reductions"]
        b5_quantity = h1_quantity
        strict_l2[quantity] = {
            "H1 deterministic": h1_quantity["H1_strict_relative_l2"],
            "B3 functional noise": b3_quantity["strict_relative_l2"],
            "B4 PDE-Refiner": b4_quantity["strict_face_contributions"][
                "ensemble_expected_paired_metrics"
            ]["relative_l2"],
            "B5 joint residual EDM": b5_quantity[
                "strict_face_contributions"
            ]["relative_l2"],
        }
        b4_separatrix = b4_quantity["separatrix_wedge"]
        h1_absolute_error = h1["transport"]["overall"]["quantities"][quantity][
            "separatrix_absolute_error"
        ]
        transport_by_quantity[quantity] = {
            "H1 deterministic": {
                "strict_relative_l2": h1_quantity["H1_strict_relative_l2"],
                "separatrix_relative_l2": h1_quantity[
                    "H1_separatrix_relative_l2"
                ],
                "separatrix_fair_crps_relative_to_H1_error": 1.0,
                "separatrix_spread_skill": None,
            },
            "B3 functional noise": {
                "strict_relative_l2": b3_quantity["strict_relative_l2"],
                "separatrix_relative_l2": b3_quantity["separatrix_relative_l2"],
                "separatrix_fair_crps_relative_to_H1_error": b3_quantity[
                    "separatrix_fair_CRPS_relative_to_parent_H1_absolute_error"
                ],
                "separatrix_spread_skill": b3_quantity[
                    "separatrix_spread_skill"
                ],
            },
            "B4 PDE-Refiner": {
                "strict_relative_l2": b4_quantity["strict_face_contributions"][
                    "ensemble_expected_paired_metrics"
                ]["relative_l2"],
                "separatrix_relative_l2": b4_separatrix[
                    "ensemble_expected_paired_metrics"
                ]["relative_l2"],
                "separatrix_fair_crps_relative_to_H1_error": b4_separatrix[
                    "ensemble_probabilistic_metrics"
                ]["fair_crps"]
                / h1_absolute_error,
                "separatrix_spread_skill": b4_separatrix[
                    "ensemble_probabilistic_metrics"
                ]["corrected_spread_skill"]["ratio"],
            },
            "B5 joint residual EDM": {
                "strict_relative_l2": b5_quantity["strict_face_contributions"][
                    "relative_l2"
                ],
                "separatrix_relative_l2": b5_quantity["separatrix_wedge"][
                    "relative_l2"
                ],
                "separatrix_fair_crps_relative_to_H1_error": b5_quantity[
                    "separatrix_wedge"
                ]["fair_crps_relative_to_H1_absolute_error"],
                "separatrix_spread_skill": b5_quantity["separatrix_wedge"][
                    "spread_skill"
                ],
            },
        }
    return {
        "models": models,
        "field_by_field": field_by_field,
        "strict_transport_relative_l2": strict_l2,
        "transport_by_quantity": transport_by_quantity,
        "important_scope_difference": (
            "B3, B4, and B5 share the one-frame one-step development task and "
            "frozen H1 parent, but use different stochastic mechanisms and losses; "
            "the comparison is descriptive rather than a single-factor ablation."
        ),
    }


def _gate_summary(gate: Mapping[str, Any]) -> dict[str, Any]:
    acceptance = gate["acceptance"]
    failed_overall: dict[str, list[dict[str, Any]]] = {}
    for family, record in acceptance["families"].items():
        failed_overall[family] = [
            {
                "name": item["name"],
                "value": item.get("value"),
                "operator": item.get("operator"),
                "lower": item.get("lower"),
                "upper": item.get("upper"),
            }
            for item in record["checks"]
            if not item["passes"]
        ]
    return {
        "status": gate["status"],
        "job_id": str(gate["slurm_job_id"]),
        "passes_complete_one_seed_gate": bool(
            acceptance["passes_complete_one_seed_gate"]
        ),
        "all_required_numeric_metrics_finite": bool(
            acceptance["all_required_numeric_metrics_finite"]
        ),
        "integrity": {
            "passes": bool(acceptance["integrity"]["passes"]),
            "check_count": int(acceptance["integrity"]["check_count"]),
            "failed_check_count": int(
                acceptance["integrity"]["failed_check_count"]
            ),
        },
        "families": gate["family_summary"],
        "failed_overall_checks": failed_overall,
        "disposition": acceptance["disposition"],
        "O3_protocol_may_be_written": bool(gate["O3_protocol_may_be_written"]),
        "O3_launch_allowed": bool(gate["O3_launch_allowed"]),
        "additional_seed_training_authorized": bool(
            acceptance["additional_seed_training_authorized"]
        ),
        "held_out_85606_access_allowed": bool(
            gate["held_out_85606_access_allowed"]
        ),
        "assimilation_allowed": bool(gate["assimilation_allowed"]),
        "diagnostic_ranking_allowed": bool(gate["diagnostic_ranking_allowed"]),
    }


def build_summary(
    *,
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    score: Mapping[str, Any],
    gate: Mapping[str, Any],
    b3: Mapping[str, Any],
    b4_gate: Mapping[str, Any],
    b4_score: Mapping[str, Any],
    h1: Mapping[str, Any],
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    for name, source in (
        ("training_result", training),
        ("evaluation_result", evaluation),
        ("score", score),
        ("final_gate", gate),
    ):
        if source.get("scope") != EXPECTED_SCOPES[name]:
            raise ValueError(f"{name} scope differs: {source.get('scope')!r}")
    for name, source in (
        ("training", training),
        ("evaluation", evaluation),
        ("score", score),
        ("gate", gate),
        ("B3", b3),
        ("B4 gate", b4_gate),
        ("B4 score", b4_score),
        ("H1 comparator", h1),
    ):
        if source.get("development_run") != "85604":
            raise ValueError(f"{name} is not restricted to development run 85604")
        if source.get("held_out_85606_read") is not False:
            raise ValueError(f"{name} does not certify that 85606 remained unread")
    if evaluation["score"]["sha256"] != input_sha256["score"]:
        raise ValueError("evaluation score identity differs")
    if gate["score_input"]["sha256"] != input_sha256["score"]:
        raise ValueError("gate score identity differs")
    if gate["training_input"]["sha256"] != input_sha256["training_result"]:
        raise ValueError("gate training-result identity differs")
    if gate["evaluation_input"]["sha256"] != input_sha256["evaluation_result"]:
        raise ValueError("gate evaluation-result identity differs")
    if evaluation["target_frames"] != [498, 624] or evaluation["target_count"] != 126:
        raise ValueError("B5 scientific evaluation target set differs")
    if evaluation["ensemble_members"] != 32:
        raise ValueError("B5 scientific ensemble size differs")
    if evaluation["truth_opened_only_after_forecast_hash"] is not True:
        raise ValueError("B5 truth separation is not certified")
    if training["config"]["absolute_time_input_allowed"] is not False:
        raise ValueError("B5 training unexpectedly allowed absolute time")
    if training["physics_derived_loss_used"] is not False:
        raise ValueError("B5 training unexpectedly used a physics-derived loss")

    field = _field_summary(score)
    spectral = _spectral_summary(score)
    transport = _transport_summary(score, h1)
    training_summary = _training_summary(training, history)
    chronology = _chronology_summary(score, b4_score, h1, gate)
    comparison = _model_comparison(
        b3, b4_score, field, spectral, transport, h1
    )
    gate_summary = _gate_summary(gate)
    return {
        "schema_version": 1,
        "scope": "phase3_B5_joint_residual_EDM_one_seed_localization_85604",
        "status": "completed_reproducible_failure_localization",
        "scientific_authority": "derived_without_rescoring_from_locked_B5_artifacts",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "training_performed_by_this_tool": False,
        "inference_performed_by_this_tool": False,
        "truth_scoring_performed_by_this_tool": False,
        "thresholds_changed_by_this_tool": False,
        "input_artifacts": {
            name: {"sha256": digest} for name, digest in input_sha256.items()
        },
        "provenance": {
            "training_job": str(training["slurm_job_id"]),
            "evaluation_job": str(evaluation["slurm_job_id"]),
            "gate_job": str(gate["slurm_job_id"]),
            "training_commit": training["paper0_commit"],
            "evaluation_commit": evaluation["paper0_commit"],
            "gate_commit": gate["gate_execution_commit"],
            "selected_checkpoint_sha256": evaluation["selected_checkpoint"][
                "sha256"
            ],
            "forecast": evaluation["forecast"],
            "score": evaluation["score"],
            "truth_opened_only_after_forecast_hash": True,
            "WandB": {
                "training": (
                    "https://wandb.ai/sdelaurentiis123-columbia-university/"
                    "tcv-diagnostics-paper0/runs/p0b5edmfull-6901531-s1701"
                ),
                "evaluation": (
                    "https://wandb.ai/sdelaurentiis123-columbia-university/"
                    "tcv-diagnostics-paper0/runs/p0b5eval-6901587-s1701"
                ),
            },
        },
        "task": {
            "context_frames": 1,
            "future_frames": 1,
            "horizon_microseconds": 3.131905426352636,
            "fields": list(FIELDS),
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "model": "frozen_H1_mean_plus_joint_field_space_EDM_residual",
            "DCAE_or_latent_representation_used_for_residual": False,
            "absolute_time_input_used": False,
            "physics_derived_training_loss_used": False,
        },
        "gates": GATES,
        "training": training_summary,
        "field_and_marginal": field,
        "spectral_and_cross_field": spectral,
        "transport": transport,
        "chronology": chronology,
        "model_comparison": comparison,
        "gate": gate_summary,
        "supported_conclusion": (
            "The B5 joint field-space residual EDM improves the H1 one-step mean, "
            "fair CRPS, and expected spectral-power recovery, but it does not learn "
            "the realization- and mode-resolved joint uncertainty required for "
            "calibrated nonlinear transport."
        ),
        "not_supported": [
            "autonomous-rollout success",
            "held-out 85606 generalization",
            "assimilation readiness",
            "diagnostic ranking",
            "steering or control",
            "a general failure of residual diffusion outside this implementation",
        ],
        "next_action": "localize B5 failure without retuning; keep O3 and 85606 closed",
    }


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale temporary output {temporary}")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    paths = {
        "training_result": args.training_result,
        "evaluation_result": args.evaluation_result,
        "history": args.history,
        "score": args.score,
        "final_gate": args.final_gate,
        "b3_result": args.b3_result,
        "b4_result": args.b4_result,
        "b4_score": args.b4_score,
        "h1_comparator": args.h1_comparator,
    }
    if "85606" in str(args.output).lower():
        raise ValueError("output path may not mention held-out 85606")
    digests = verify_inputs(paths)
    result = build_summary(
        training=load_json(args.training_result),
        evaluation=load_json(args.evaluation_result),
        history=load_history(args.history),
        score=load_json(args.score),
        gate=load_json(args.final_gate),
        b3=load_json(args.b3_result),
        b4_gate=load_json(args.b4_result),
        b4_score=load_json(args.b4_score),
        h1=load_json(args.h1_comparator),
        input_sha256=digests,
    )
    write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve(strict=True)),
                "sha256": sha256_path(args.output.resolve(strict=True)),
                "status": result["status"],
                "gate_passes": result["gate"]["passes_complete_one_seed_gate"],
                "held_out_85606_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
