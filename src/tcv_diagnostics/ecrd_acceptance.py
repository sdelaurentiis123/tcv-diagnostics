"""Prospectively frozen three-seed ECRD acceptance reduction.

All inputs are already-computed 85604 score dictionaries.  This module opens
no data or forecasts and implements the exact reductions recorded in
``ECRD_EVALUATION_IMPLEMENTATION_FREEZE.md``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .b2_field_metrics import B2_FIELDS
from .b2_probabilistic_metrics import moving_block_bootstrap_indices
from .b2_spectral_metrics import B2_MODE_BANDS
from .b5_covariance_localization import B5_FINITE_MEMBER_FACTOR
from .codec_transport import TRANSPORT_QUANTITIES
from .ecrd_scoring import ECRD_EVALUATION_BLOCKS, ECRD_EVALUATION_TARGETS
from .ecrd_training import ECRD_ARMS, ECRD_MODEL_SEEDS


ECRD_BOOTSTRAP_BLOCK_LENGTH = 12
ECRD_BOOTSTRAP_REPLICATES = 2_000
ECRD_BOOTSTRAP_SEED = 85_604_351
ECRD_MATERIAL_POWER_RANGE = (0.75, 1.30)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _seed_records(
    records: Mapping[int | str, Mapping[str, Any]], *, arm: str
) -> dict[int, Mapping[str, Any]]:
    converted = {int(seed): score for seed, score in records.items()}
    if tuple(sorted(converted)) != ECRD_MODEL_SEEDS:
        raise ValueError(f"{arm} must contain exactly three model seeds")
    for seed, score in converted.items():
        if (
            score.get("scope")
            != "ECRD_truth_separated_probabilistic_scoring_85604"
            or score.get("development_run") != "85604"
            or score.get("held_out_85606_read") is not False
            or score.get("guard_frames_read") is not False
            or score.get("physics_derived_training_loss_used") is not False
            or score.get("target_truth_used_during_forecast_generation") is not False
            or score.get("arm") != arm
            or score.get("model_seed") != seed
            or score.get("target_frames") != [498, 624]
            or score.get("target_count") != 126
            or score.get("validation_blocks")
            != {
                name: [block[0], block[-1] + 1]
                for name, block in ECRD_EVALUATION_BLOCKS.items()
            }
        ):
            raise ValueError(f"{arm} seed {seed} score contract differs")
    return converted


def validate_ecrd_score_matrix(
    scores: Mapping[str, Mapping[int | str, Mapping[str, Any]]]
) -> dict[str, dict[int, Mapping[str, Any]]]:
    if tuple(scores) != ECRD_ARMS:
        raise ValueError("ECRD score matrix arm order differs")
    return {arm: _seed_records(scores[arm], arm=arm) for arm in ECRD_ARMS}


def _field_block(score: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    records = score["field_and_marginal_calibration"][
        "chronological_blocks_eligible_union"
    ]
    index = tuple(ECRD_EVALUATION_BLOCKS).index(name)
    record = records[index]
    block = ECRD_EVALUATION_BLOCKS[name]
    if record.get("target_frames") != [block[0], block[-1] + 1]:
        raise ValueError(f"field block {name} identity differs")
    return record


def _spectral_root(
    score: Mapping[str, Any], block_name: str | None
) -> Mapping[str, Any]:
    root = score["spectral_and_cross_field"]
    if block_name is None:
        return root["overall"]
    records = root["chronological_blocks"]
    selected = [record for record in records if record.get("name") == block_name]
    if len(selected) != 1:
        raise ValueError(f"spectral block {block_name} identity differs")
    return selected[0]["score"]


def _complex_curve(record: Mapping[str, Any], name: str) -> np.ndarray:
    values = record[name]
    real = np.asarray(values["real"], dtype=np.float64)
    imaginary = np.asarray(values["imag"], dtype=np.float64)
    if real.shape != imaginary.shape or real.ndim != 1 or real.size < 8:
        raise ValueError(f"complex curve {name} shape differs")
    result = real + 1j * imaginary
    if not np.all(np.isfinite(result)):
        raise ValueError(f"complex curve {name} contains non-finite values")
    return result


def _real_curve(record: Mapping[str, Any], name: str) -> np.ndarray:
    values = np.asarray(record[name], dtype=np.float64)
    if values.ndim != 1 or values.size < 8 or not np.all(np.isfinite(values)):
        raise ValueError(f"real curve {name} differs")
    return values


def _pooled_field_spread_skill(
    records: Mapping[int, Mapping[str, Any]], field: str
) -> float:
    channel = B2_FIELDS.index(field)
    squared = 0.0
    variance = 0.0
    count = 0
    for score in records.values():
        per_target = score["field_and_marginal_calibration"][
            "per_target_eligible_union_sufficient_statistics"
        ]
        if len(per_target) != len(ECRD_EVALUATION_TARGETS):
            raise ValueError("ECRD field per-target record count differs")
        for expected, target_record in zip(ECRD_EVALUATION_TARGETS, per_target):
            if target_record.get("target_frame") != expected:
                raise ValueError("ECRD field per-target order differs")
            item = target_record["fields"][field]
            squared += _finite(item["squared_error_sum"], "field squared error")
            variance += _finite(item["member_variance_sum"], "field variance")
            count += int(item["count"])
        if channel < 0:  # pragma: no cover - defensive field-order assertion
            raise AssertionError("field index is invalid")
    if count <= 0 or squared <= 0.0 or variance <= 0.0:
        raise ValueError("ECRD pooled field spread-skill inputs differ")
    return math.sqrt(B5_FINITE_MEMBER_FACTOR * variance / squared)


def _material_power_summary(
    records: Mapping[int, Mapping[str, Any]],
    materiality: Mapping[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for field in B2_FIELDS:
        for label, _, _ in B2_MODE_BANDS:
            if not bool(materiality["fields"][field]["bands"][label]["material"]):
                continue
            ratios = [
                _positive(
                    _spectral_root(score, None)["toroidal_field_power"][field][
                        "bands"
                    ][label]["member_expected_power_ratio"],
                    f"{field} {label} power ratio",
                )
                for score in records.values()
            ]
            ratio = float(np.mean(ratios))
            checks.append(
                {
                    "field": field,
                    "band": label,
                    "seed_mean_power_ratio": ratio,
                    "absolute_log_error": abs(math.log(ratio)),
                    "passes_inherited_range": (
                        ECRD_MATERIAL_POWER_RANGE[0]
                        <= ratio
                        <= ECRD_MATERIAL_POWER_RANGE[1]
                    ),
                }
            )
    if not checks:
        raise ValueError("ECRD material power set is empty")
    return {
        "material_check_count": len(checks),
        "passing_count": sum(item["passes_inherited_range"] for item in checks),
        "median_absolute_log_power_ratio_error": float(
            np.median([item["absolute_log_error"] for item in checks])
        ),
        "checks": checks,
    }


def _cross_summary(
    records: Mapping[int, Mapping[str, Any]], block_name: str | None
) -> dict[str, float]:
    truth_crosses = []
    forecast_crosses = []
    truth_ne_power = []
    truth_phi_power = []
    forecast_ne_power = []
    forecast_phi_power = []
    for score in records.values():
        root = _spectral_root(score, block_name)
        cross_curves = root["toroidal_cross_field"]["Ne-phi"]["curves"]
        truth_crosses.append(_complex_curve(cross_curves, "truth_cross_spectrum"))
        forecast_crosses.append(
            _complex_curve(cross_curves, "member_expected_cross_spectrum")
        )
        truth_ne_power.append(
            _real_curve(
                root["toroidal_field_power"]["Ne"]["curves"], "truth_power"
            )
        )
        truth_phi_power.append(
            _real_curve(
                root["toroidal_field_power"]["phi"]["curves"], "truth_power"
            )
        )
        forecast_ne_power.append(
            _real_curve(
                root["toroidal_field_power"]["Ne"]["curves"],
                "member_expected_power",
            )
        )
        forecast_phi_power.append(
            _real_curve(
                root["toroidal_field_power"]["phi"]["curves"],
                "member_expected_power",
            )
        )
    truth_cross = np.mean(truth_crosses, axis=0)
    forecast_cross = np.mean(forecast_crosses, axis=0)
    truth_ne = np.mean(truth_ne_power, axis=0)
    truth_phi = np.mean(truth_phi_power, axis=0)
    forecast_ne = np.mean(forecast_ne_power, axis=0)
    forecast_phi = np.mean(forecast_phi_power, axis=0)
    selected = slice(1, 8)
    truth = truth_cross[selected]
    forecast = forecast_cross[selected]
    weight = np.abs(truth)
    denominator = float(np.sum(weight))
    if denominator <= 0.0:
        raise ValueError("ECRD Ne-phi truth cross amplitude is zero")
    truth_coherence = np.abs(truth) ** 2 / (
        truth_ne[selected] * truth_phi[selected]
    )
    forecast_coherence = np.abs(forecast) ** 2 / (
        forecast_ne[selected] * forecast_phi[selected]
    )
    if not np.all(np.isfinite(truth_coherence)) or not np.all(
        np.isfinite(forecast_coherence)
    ):
        raise ValueError("ECRD Ne-phi coherence is non-finite")
    phase_error = np.abs(np.angle(forecast * np.conjugate(truth)))
    return {
        "complex_cross_spectrum_relative_L1_error": float(
            np.sum(np.abs(forecast - truth)) / denominator
        ),
        "truth_amplitude_weighted_absolute_coherence_error": float(
            np.sum(weight * np.abs(forecast_coherence - truth_coherence))
            / denominator
        ),
        "truth_amplitude_weighted_absolute_phase_error_degrees": float(
            math.degrees(np.sum(weight * phase_error) / denominator)
        ),
    }


def _transport_series(
    score: Mapping[str, Any], quantity: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = score["memberwise_transport"]["overall"]["quantities"][quantity][
        "separatrix_time_series"
    ]
    targets = np.asarray(series["target_frame"], dtype=np.int64)
    truth = np.asarray(series["truth"], dtype=np.float64)
    mean = np.asarray(series["ensemble_expected"], dtype=np.float64)
    standard_deviation = np.asarray(
        series["member_standard_deviation_ddof1"], dtype=np.float64
    )
    if (
        not np.array_equal(targets, ECRD_EVALUATION_TARGETS)
        or truth.shape != targets.shape
        or mean.shape != targets.shape
        or standard_deviation.shape != targets.shape
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(standard_deviation))
        or np.any(standard_deviation < 0.0)
    ):
        raise ValueError(f"ECRD {quantity} separatrix series differs")
    return truth, mean, standard_deviation**2


def _integrated_ratios(
    records: Mapping[int, Mapping[str, Any]],
    indices: np.ndarray | None = None,
) -> dict[str, float]:
    selected = np.arange(126, dtype=np.int64) if indices is None else np.asarray(indices)
    if selected.ndim != 1 or selected.size < 1:
        raise ValueError("ECRD integrated selected indices differ")
    ratios: dict[str, float] = {}
    for quantity in TRANSPORT_QUANTITIES:
        variance_sum = 0.0
        squared_error_sum = 0.0
        scalar_count = 0
        for score in records.values():
            truth, mean, variance = _transport_series(score, quantity)
            variance_sum += float(np.sum(variance[selected]))
            squared_error_sum += float(np.sum((mean[selected] - truth[selected]) ** 2))
            scalar_count += int(selected.size)
        if scalar_count <= 0 or squared_error_sum <= 0.0:
            raise ValueError("ECRD integrated spread-skill denominator differs")
        ratios[quantity] = math.sqrt(
            B5_FINITE_MEMBER_FACTOR * variance_sum / squared_error_sum
        )
    return ratios


def _local_ratios(records: Mapping[int, Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for quantity in TRANSPORT_QUANTITIES:
        predicted = 0.0
        realized = 0.0
        for score in records.values():
            values = score["spatial_transport_covariance"]["overall"][
                "local_covariance"
            ]["quantities"][quantity]["covariance_decomposition"]
            predicted += _finite(
                values["ensemble_diagonal_variance_mean_over_targets"],
                "local predicted variance",
            )
            realized += _finite(
                values["error_diagonal_MSE_sum_mean_over_targets"],
                "local realized squared error",
            )
        if predicted < 0.0 or realized <= 0.0:
            raise ValueError("ECRD local spread-skill moments differ")
        result[quantity] = math.sqrt(B5_FINITE_MEMBER_FACTOR * predicted / realized)
    return result


def _spatial_covariance_errors(
    records: Mapping[int, Mapping[str, Any]]
) -> dict[str, float]:
    result = {}
    for quantity in TRANSPORT_QUANTITIES:
        values = [
            _finite(
                score["spatial_transport_covariance"]["overall"][
                    "full_spatial_covariance_sketch"
                ]["quantities"][quantity]["relative_frobenius_error_sketch"],
                "spatial covariance error",
            )
            for score in records.values()
        ]
        result[quantity] = float(np.mean(values))
    return result


def _overall_fair_crps(score: Mapping[str, Any]) -> float:
    return _positive(
        score["field_and_marginal_calibration"]["regions"]["eligible_union"][
            "aggregate"
        ]["equal_channel_fair_crps"],
        "overall field fair CRPS",
    )


def summarize_ecrd_arm(
    records: Mapping[int, Mapping[str, Any]],
    *,
    materiality: Mapping[str, Any],
) -> dict[str, Any]:
    block_fair = {
        name: float(
            np.mean(
                [
                    _positive(
                        _field_block(score, name)["aggregate"][
                            "equal_channel_fair_crps"
                        ],
                        f"{name} fair CRPS",
                    )
                    for score in records.values()
                ]
            )
        )
        for name in ECRD_EVALUATION_BLOCKS
    }
    field_ratios = {
        field: _pooled_field_spread_skill(records, field) for field in B2_FIELDS
    }
    integrated_ratios = _integrated_ratios(records)
    local_ratios = _local_ratios(records)
    spatial_errors = _spatial_covariance_errors(records)
    return _json_safe(
        {
            "seed_mean_overall_equal_field_fair_crps": float(
                np.mean([_overall_fair_crps(score) for score in records.values()])
            ),
            "chronological_block_equal_field_fair_crps": block_fair,
            "field_spread_skill_ratio": field_ratios,
            "median_absolute_log_field_spread_skill_error": float(
                np.median([abs(math.log(value)) for value in field_ratios.values()])
            ),
            "material_spectral_power": _material_power_summary(
                records, materiality
            ),
            "Ne_phi_dependence": {
                "overall": _cross_summary(records, None),
                "chronological_blocks": {
                    name: _cross_summary(records, name)
                    for name in ECRD_EVALUATION_BLOCKS
                },
            },
            "spatial_transport_covariance_relative_error": spatial_errors,
            "median_spatial_transport_covariance_relative_error": float(
                np.median(list(spatial_errors.values()))
            ),
            "integrated_transport_spread_skill_ratio": integrated_ratios,
            "median_absolute_log_integrated_transport_spread_skill_error": float(
                np.median(
                    [abs(math.log(value)) for value in integrated_ratios.values()]
                )
            ),
            "median_integrated_transport_spread_skill_ratio": float(
                np.median(list(integrated_ratios.values()))
            ),
            "local_transport_spread_skill_ratio": local_ratios,
        }
    )


def _per_seed_integrated_error(score: Mapping[str, Any]) -> float:
    ratios = _integrated_ratios({int(score["model_seed"]): score})
    return float(np.median([abs(math.log(value)) for value in ratios.values()]))


def _bootstrap_fair_crps(
    records: Mapping[int, Mapping[str, Any]], indices: np.ndarray
) -> float:
    seed_values = []
    for score in records.values():
        per_target = score["field_and_marginal_calibration"][
            "per_target_eligible_union_sufficient_statistics"
        ]
        fields = []
        for field in B2_FIELDS:
            fair = np.asarray(
                [item["fields"][field]["fair_crps_sum"] for item in per_target],
                dtype=np.float64,
            )
            count = np.asarray(
                [item["fields"][field]["count"] for item in per_target],
                dtype=np.float64,
            )
            fields.append(float(np.sum(fair[indices]) / np.sum(count[indices])))
        seed_values.append(float(np.mean(fields)))
    return float(np.mean(seed_values))


def _paired_bootstrap(
    candidate: Mapping[int, Mapping[str, Any]],
    reference: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    bootstrap = moving_block_bootstrap_indices(
        126,
        block_length=ECRD_BOOTSTRAP_BLOCK_LENGTH,
        replicates=ECRD_BOOTSTRAP_REPLICATES,
        seed=ECRD_BOOTSTRAP_SEED,
    )
    fair_improvement = np.empty(ECRD_BOOTSTRAP_REPLICATES, dtype=np.float64)
    integrated_improvement = np.empty(ECRD_BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate, indices in enumerate(bootstrap):
        reference_fair = _bootstrap_fair_crps(reference, indices)
        candidate_fair = _bootstrap_fair_crps(candidate, indices)
        fair_improvement[replicate] = (
            reference_fair - candidate_fair
        ) / reference_fair
        reference_ratios = _integrated_ratios(reference, indices)
        candidate_ratios = _integrated_ratios(candidate, indices)
        reference_error = float(
            np.median([abs(math.log(value)) for value in reference_ratios.values()])
        )
        candidate_error = float(
            np.median([abs(math.log(value)) for value in candidate_ratios.values()])
        )
        integrated_improvement[replicate] = (
            reference_error - candidate_error
        ) / reference_error

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "median": float(np.quantile(values, 0.5, method="linear")),
            "lower_2p5": float(np.quantile(values, 0.025, method="linear")),
            "upper_97p5": float(np.quantile(values, 0.975, method="linear")),
        }

    return {
        "method": "paired_noncircular_moving_block_bootstrap",
        "block_length_frames": ECRD_BOOTSTRAP_BLOCK_LENGTH,
        "replicates": ECRD_BOOTSTRAP_REPLICATES,
        "seed": ECRD_BOOTSTRAP_SEED,
        "conditional_on_single_85604_run": True,
        "candidate_minus_B5_relative_improvement": {
            "equal_field_fair_CRPS": interval(fair_improvement),
            "integrated_transport_calibration_error": interval(
                integrated_improvement
            ),
        },
    }


def evaluate_ecrd_candidate(
    candidate: Mapping[int, Mapping[str, Any]],
    reference: Mapping[int, Mapping[str, Any]],
    *,
    materiality: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_summary = summarize_ecrd_arm(candidate, materiality=materiality)
    reference_summary = summarize_ecrd_arm(reference, materiality=materiality)

    block_improvements = {
        name: 1.0
        - candidate_summary["chronological_block_equal_field_fair_crps"][name]
        / reference_summary["chronological_block_equal_field_fair_crps"][name]
        for name in ECRD_EVALUATION_BLOCKS
    }
    candidate_field_error = candidate_summary[
        "median_absolute_log_field_spread_skill_error"
    ]
    reference_field_error = reference_summary[
        "median_absolute_log_field_spread_skill_error"
    ]
    family1 = all(value >= 0.02 for value in block_improvements.values()) and (
        candidate_field_error <= 0.90 * reference_field_error
    )

    candidate_power = candidate_summary["material_spectral_power"]
    reference_power = reference_summary["material_spectral_power"]
    family2 = (
        candidate_power["median_absolute_log_power_ratio_error"]
        <= 1.05 * reference_power["median_absolute_log_power_ratio_error"]
        and candidate_power["passing_count"] >= reference_power["passing_count"]
    )

    candidate_cross = candidate_summary["Ne_phi_dependence"]
    reference_cross = reference_summary["Ne_phi_dependence"]
    cross_key = "complex_cross_spectrum_relative_L1_error"
    coherence_key = "truth_amplitude_weighted_absolute_coherence_error"
    phase_key = "truth_amplitude_weighted_absolute_phase_error_degrees"
    cross_blocks_improved = sum(
        candidate_cross["chronological_blocks"][name][cross_key]
        < reference_cross["chronological_blocks"][name][cross_key]
        for name in ECRD_EVALUATION_BLOCKS
    )
    coherence_blocks_improved = sum(
        candidate_cross["chronological_blocks"][name][coherence_key]
        < reference_cross["chronological_blocks"][name][coherence_key]
        for name in ECRD_EVALUATION_BLOCKS
    )
    family3 = (
        candidate_cross["overall"][cross_key]
        <= 0.90 * reference_cross["overall"][cross_key]
        and candidate_cross["overall"][coherence_key]
        <= 0.90 * reference_cross["overall"][coherence_key]
        and cross_blocks_improved >= 2
        and coherence_blocks_improved >= 2
        and candidate_cross["overall"][phase_key]
        <= reference_cross["overall"][phase_key] + 2.0
    )

    candidate_spatial = candidate_summary[
        "spatial_transport_covariance_relative_error"
    ]
    reference_spatial = reference_summary[
        "spatial_transport_covariance_relative_error"
    ]
    spatial_quantities_improved = sum(
        candidate_spatial[name] < reference_spatial[name]
        for name in TRANSPORT_QUANTITIES
    )
    family4 = (
        candidate_summary["median_spatial_transport_covariance_relative_error"]
        <= 0.85
        * reference_summary["median_spatial_transport_covariance_relative_error"]
        and spatial_quantities_improved >= 3
    )

    candidate_integrated = candidate_summary[
        "integrated_transport_spread_skill_ratio"
    ]
    reference_integrated = reference_summary[
        "integrated_transport_spread_skill_ratio"
    ]
    ratios_moving_toward_one = sum(
        reference_integrated[name] < 1.0
        and candidate_integrated[name] - reference_integrated[name] >= 0.10
        and abs(candidate_integrated[name] - 1.0)
        < abs(reference_integrated[name] - 1.0)
        for name in TRANSPORT_QUANTITIES
    )
    family5 = (
        candidate_summary[
            "median_absolute_log_integrated_transport_spread_skill_error"
        ]
        <= 0.80
        * reference_summary[
            "median_absolute_log_integrated_transport_spread_skill_error"
        ]
        and ratios_moving_toward_one >= 3
        and candidate_summary["median_integrated_transport_spread_skill_ratio"]
        >= 0.60
    )

    local = candidate_summary["local_transport_spread_skill_ratio"]
    family6 = all(value <= 1.25 for value in local.values()) and sum(
        0.80 <= value <= 1.25 for value in local.values()
    ) >= 3

    robust_seeds = []
    for seed in ECRD_MODEL_SEEDS:
        fair_better = _overall_fair_crps(candidate[seed]) < _overall_fair_crps(
            reference[seed]
        )
        integrated_better = _per_seed_integrated_error(
            candidate[seed]
        ) < _per_seed_integrated_error(reference[seed])
        if fair_better and integrated_better:
            robust_seeds.append(seed)
    bootstrap = _paired_bootstrap(candidate, reference)
    bootstrap_values = bootstrap["candidate_minus_B5_relative_improvement"]
    family7 = (
        len(robust_seeds) >= 2
        and bootstrap_values["equal_field_fair_CRPS"]["lower_2p5"] > 0.0
        and bootstrap_values["integrated_transport_calibration_error"][
            "lower_2p5"
        ]
        > 0.0
    )

    families = {
        "1_marginal_forecast": {
            "passes": bool(family1),
            "block_relative_fair_CRPS_improvement": block_improvements,
            "candidate_median_absolute_log_spread_skill_error": candidate_field_error,
            "B5_median_absolute_log_spread_skill_error": reference_field_error,
        },
        "2_spectral_retention": {
            "passes": bool(family2),
            "candidate": candidate_power,
            "B5": reference_power,
        },
        "3_Ne_phi_dependence": {
            "passes": bool(family3),
            "candidate": candidate_cross,
            "B5": reference_cross,
            "cross_spectrum_blocks_improved": cross_blocks_improved,
            "coherence_blocks_improved": coherence_blocks_improved,
        },
        "4_spatial_transport_covariance": {
            "passes": bool(family4),
            "candidate": candidate_spatial,
            "B5": reference_spatial,
            "quantities_improved": spatial_quantities_improved,
        },
        "5_integrated_transport_spread": {
            "passes": bool(family5),
            "candidate_ratios": candidate_integrated,
            "B5_ratios": reference_integrated,
            "ratios_moving_at_least_0p10_toward_one": ratios_moving_toward_one,
        },
        "6_no_local_overdispersion": {
            "passes": bool(family6),
            "candidate_local_ratios": local,
            "ratios_in_0p80_to_1p25": sum(
                0.80 <= value <= 1.25 for value in local.values()
            ),
        },
        "7_robustness": {
            "passes": bool(family7),
            "same_seed_joint_improvement": robust_seeds,
            "paired_bootstrap": bootstrap,
        },
    }
    return _json_safe(
        {
            "candidate_summary": candidate_summary,
            "B5_summary": reference_summary,
            "families": families,
            "all_seven_families_pass": all(
                bool(record["passes"]) for record in families.values()
            ),
        }
    )


def evaluate_ecrd_model_ladder(
    scores: Mapping[str, Mapping[int | str, Mapping[str, Any]]],
    *,
    spectral_materiality: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate both eligible ECRD arms and apply the simplicity rule."""

    matrix = validate_ecrd_score_matrix(scores)
    summaries = {
        arm: summarize_ecrd_arm(records, materiality=spectral_materiality)
        for arm, records in matrix.items()
    }
    candidates = {
        arm: evaluate_ecrd_candidate(
            matrix[arm], matrix["B5"], materiality=spectral_materiality
        )
        for arm in ("ECRD", "ECRD-History")
    }
    passing = [
        arm for arm, record in candidates.items() if record["all_seven_families_pass"]
    ]
    if passing == ["ECRD", "ECRD-History"]:
        simple_error = summaries["ECRD"][
            "median_absolute_log_integrated_transport_spread_skill_error"
        ]
        history_error = summaries["ECRD-History"][
            "median_absolute_log_integrated_transport_spread_skill_error"
        ]
        selected = "ECRD-History" if history_error <= 0.90 * simple_error else "ECRD"
    elif len(passing) == 1:
        selected = passing[0]
    else:
        selected = None
    return _json_safe(
        {
            "schema_version": 1,
            "scope": "ECRD_three_seed_model_development_acceptance_85604",
            "development_run": "85604",
            "held_out_85606_read": False,
            "physics_derived_training_loss_used": False,
            "model_score_substitution_used": False,
            "summaries": summaries,
            "eligible_candidate_gates": candidates,
            "passing_arms": passing,
            "selected_arm": selected,
            "held_out_release_eligible": selected is not None,
            "held_out_85606_access_authorized_by_this_record": False,
            "assimilation_authorized": False,
            "diagnostic_ranking_authorized": False,
            "failure_next_action": (
                None
                if selected is not None
                else "construct_exact_evolved_Hermes_state_and_request_independent_restarts"
            ),
        }
    )
