"""B4 scoring with a locked B2 final-stage engine and explicit stage repair.

The final M32 artifact delegates unchanged numerical metrics to the byte-locked
B2 engine.  The separate M4 stage artifact is used only for the prospectively
defined B4 H-det repair comparison across levels zero through three.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .b2_field_metrics import B2_FIELDS, gauge_fix_phi_channel
from .b2_scoring import (
    score_b2_forecast,
    score_b2_forecast_smoke,
    validate_b2_spectral_materiality,
)
from .b2_spectral_metrics import B2_CROSS_PAIRS, B2_MODE_BANDS
from .codec_training import sha256_path
from .codec_transport import (
    TRANSPORT_QUANTITIES,
    CodecTransportGeometry,
    direct_pressure_transport_state,
    evaluate_transport_state,
)
from .matched_o1_transport import NativeTruthCatalog
from .model_training_data import FAMILY_FIELDS, ModelDatasetCatalog
from .o2_training_data import OneStepWindowDataset, strict_o2_targets
from .resampling import periodic_resample_float32


ROOT = Path(__file__).resolve().parents[2]
LOCKED_METRIC_SOURCES = {
    "src/tcv_diagnostics/b2_probabilistic_metrics.py": (
        "edef6fbbe7b40348fa450c7428d796f4b5ebc3d9b2070e135c7bb3f58a2b6650"
    ),
    "src/tcv_diagnostics/b2_field_metrics.py": (
        "c2d0f5e764b783f7a6a240fbd3f11f6c0a4fd52a173d9f1dd1eb97ccff62a0db"
    ),
    "src/tcv_diagnostics/b2_spectral_metrics.py": (
        "382fc683519d01185d0e5314196cd0c62f5e39e60f5e1aa06478e74acda8761e"
    ),
    "src/tcv_diagnostics/b2_transport_metrics.py": (
        "b78ea33f641fe6409ca5a55503f3729013f2da3cc78f93671f63c6fadafcb02e"
    ),
    "src/tcv_diagnostics/b2_scoring.py": (
        "2dfdf6f7b620302826971c9fec4ed8233f46fa1950c8461ed9d79194411178fe"
    ),
    "src/tcv_diagnostics/geometry.py": (
        "4f5eda7001bf9b42cefb224842a1dee4a955028a1aa063a57db6c447879f424c"
    ),
    "src/tcv_diagnostics/codec_transport.py": (
        "201a9628564b1ad5e476cbee52edf5eac458c61dadc1c7057a5b6e205de46d45"
    ),
}


def verify_locked_b4_metric_sources() -> dict[str, str]:
    """Fail closed if any numerical metric source changed after protocol freeze."""

    verified: dict[str, str] = {}
    for relative, expected in LOCKED_METRIC_SOURCES.items():
        actual = sha256_path(ROOT / relative)
        if actual != expected:
            raise RuntimeError(
                f"frozen B4 metric source differs for {relative}: {actual}"
            )
        verified[relative] = actual
    return verified


def _validate_artifact_identity(forecast_artifact: Any, *, stages: bool) -> None:
    metadata = forecast_artifact.metadata
    if (
        forecast_artifact.model_seed != 1701
        or metadata.get("source_kind") != "selected_B4_PDE_Refiner"
        or metadata.get("arm") != "B4-PDE-Refiner-H1"
        or metadata.get("seed") != 1701
        or metadata.get("context_frames") != 1
        or metadata.get("target_truth_read") is not False
        or metadata.get("absolute_time_input") is not False
        or metadata.get("member_prefixes_regenerated") is not False
        or metadata.get("posthoc_calibration") is not False
    ):
        kind = "stage" if stages else "final"
        raise ValueError(f"B4 {kind} forecast identity differs")


def _relabel_final_score(
    score: Mapping[str, Any],
    *,
    bounded_smoke: bool,
    verified_sources: Mapping[str, str],
) -> dict[str, Any]:
    expected_scope = (
        "bounded_non_scientific_B2_evaluator_smoke_scoring_85604"
        if bounded_smoke
        else "B2_truth_separated_probabilistic_scoring_85604"
    )
    if (
        score.get("scope") != expected_scope
        or score.get("development_run") != "85604"
        or score.get("held_out_85606_read") is not False
        or score.get(
            "truth_opened_only_after_forecast_was_closed_and_hash_verified"
        )
        is not True
    ):
        raise RuntimeError("delegated frozen B4 metric-engine result differs")
    result = dict(score)
    result["scope"] = (
        "bounded_non_scientific_B4_PDE_Refiner_H1_final_scoring_85604"
        if bounded_smoke
        else "B4_PDE_Refiner_H1_final_M32_scoring_85604"
    )
    result["model_arm"] = "B4-PDE-Refiner-H1"
    result["context_frames"] = 1
    result["refinement_stage"] = 3
    result["metric_engine"] = {
        "identity": "byte_locked_B2_numerical_metric_engine",
        "numerical_definitions_changed_for_B4_final": False,
        "original_delegated_scope": expected_scope,
        "source_sha256": dict(verified_sources),
    }
    return result


def _score_final(
    *,
    catalog: Any,
    forecast_artifact: Any,
    native_truth: Any,
    geometry: Any,
    event_threshold_record: Mapping[str, Any],
    target_frames: Sequence[int],
    bounded_smoke: bool,
) -> dict[str, Any]:
    sources = verify_locked_b4_metric_sources()
    _validate_artifact_identity(forecast_artifact, stages=False)
    scorer = score_b2_forecast_smoke if bounded_smoke else score_b2_forecast
    score = scorer(
        catalog=catalog,
        forecast_artifact=forecast_artifact,
        native_truth=native_truth,
        geometry=geometry,
        event_threshold_record=event_threshold_record,
        target_frames=target_frames,
        model_seed=1701,
    )
    return _relabel_final_score(
        score, bounded_smoke=bounded_smoke, verified_sources=sources
    )


def score_pde_refiner_final(
    *,
    catalog: Any,
    forecast_artifact: Any,
    native_truth: Any,
    geometry: Any,
    event_threshold_record: Mapping[str, Any],
    target_frames: Sequence[int],
) -> dict[str, Any]:
    """Score the full B4 level-three M32 artifact with unchanged B2 metrics."""

    return _score_final(
        catalog=catalog,
        forecast_artifact=forecast_artifact,
        native_truth=native_truth,
        geometry=geometry,
        event_threshold_record=event_threshold_record,
        target_frames=target_frames,
        bounded_smoke=False,
    )


def score_pde_refiner_final_smoke(
    *,
    catalog: Any,
    forecast_artifact: Any,
    native_truth: Any,
    geometry: Any,
    event_threshold_record: Mapping[str, Any],
    target_frames: Sequence[int],
) -> dict[str, Any]:
    """Run the identical final scorer on four targets as a preflight."""

    return _score_final(
        catalog=catalog,
        forecast_artifact=forecast_artifact,
        native_truth=native_truth,
        geometry=geometry,
        event_threshold_record=event_threshold_record,
        target_frames=target_frames,
        bounded_smoke=True,
    )


def decode_stage_member_forecasts(
    catalog: ModelDatasetCatalog,
    standardized_forecast: np.ndarray,
) -> np.ndarray:
    """Inverse-transform one M4 stage without clipping or member reduction."""

    values = np.asarray(standardized_forecast)
    if values.shape != (4, len(B2_FIELDS), 64, 32, 88):
        raise ValueError("B4 standardized stage forecast shape differs")
    if np.iscomplexobj(values) or not np.issubdtype(values.dtype, np.number):
        raise TypeError("B4 standardized stage forecast must be real numeric")
    if not np.all(np.isfinite(values)):
        raise ValueError("B4 standardized stage forecast contains non-finite values")
    physical = np.stack(
        [
            catalog.normalization.records[field].decode(values[:, channel])
            for channel, field in enumerate(FAMILY_FIELDS["c5p"])
        ],
        axis=1,
    )
    if physical.shape != (4, len(B2_FIELDS), 64, 32, 88):
        raise RuntimeError("B4 decoded physical stage shape differs")
    if not np.all(np.isfinite(physical)):
        raise ValueError("B4 decoded physical stage is non-finite")
    return np.asarray(physical, dtype=np.float64)


def _one_sided_weights(size: int) -> np.ndarray:
    weights = np.ones(int(size) // 2 + 1, dtype=np.float64)
    if int(size) % 2 == 0:
        weights[1:-1] = 2.0
    else:
        weights[1:] = 2.0
    return weights


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    first, second = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    result = np.full(first.shape, np.nan, dtype=np.float64)
    np.divide(first, second, out=result, where=second > 0.0)
    return result


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    value = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(value) & np.isfinite(weight) & (weight >= 0.0)
    denominator = float(np.sum(weight[valid], dtype=np.float64))
    if denominator <= 0.0:
        return math.nan
    return float(
        np.sum(value[valid] * weight[valid], dtype=np.float64) / denominator
    )


def _memberwise_separatrix_transport(
    *,
    physical_forecast_model88: np.ndarray,
    native_truth: Mapping[str, np.ndarray],
    geometry: CodecTransportGeometry,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    forecast = np.asarray(physical_forecast_model88, dtype=np.float64)
    if forecast.shape != (4, 5, 64, 32, 88) or not np.all(np.isfinite(forecast)):
        raise ValueError("B4 physical stage must have finite shape [4,5,64,32,88]")
    if set(native_truth) != {"Ne", "Pe", "Pi", "phi"}:
        raise ValueError("B4 native transport truth fields differ")
    truth = {
        field: np.asarray(values, dtype=np.float64)
        for field, values in native_truth.items()
    }
    if any(
        values.shape != (1, 64, 32, 81) or not np.all(np.isfinite(values))
        for values in truth.values()
    ):
        raise ValueError("B4 native transport truth shape or values differ")
    native_forecast = periodic_resample_float32(
        forecast[:, :4], 81, axis=-1
    ).astype(np.float64)
    truth_state = direct_pressure_transport_state(
        truth["Ne"], truth["Pe"], truth["Pi"], truth["phi"]
    )
    truth_evaluated = evaluate_transport_state(truth_state, geometry)
    truth_outputs = {
        quantity: float(truth_evaluated[quantity]["separatrix_wedge"][0])
        for quantity in TRANSPORT_QUANTITIES
    }
    member_outputs = {quantity: [] for quantity in TRANSPORT_QUANTITIES}
    for member in range(4):
        state = direct_pressure_transport_state(
            native_forecast[member, 0][None],
            native_forecast[member, 1][None],
            native_forecast[member, 2][None],
            native_forecast[member, 3][None],
        )
        evaluated = evaluate_transport_state(state, geometry)
        for quantity in TRANSPORT_QUANTITIES:
            member_outputs[quantity].append(
                float(evaluated[quantity]["separatrix_wedge"][0])
            )
    forecast_outputs = {
        quantity: np.asarray(member_outputs[quantity], dtype=np.float64)
        for quantity in TRANSPORT_QUANTITIES
    }
    if any(
        values.shape != (4,) or not np.all(np.isfinite(values))
        for values in forecast_outputs.values()
    ) or any(not math.isfinite(value) for value in truth_outputs.values()):
        raise FloatingPointError("B4 stage transport output is non-finite")
    return forecast_outputs, truth_outputs


class _StageRepairAccumulator:
    """Sufficient statistics for the four prospectively frozen H-det errors."""

    def __init__(self, *, eligible_xy_mask: np.ndarray) -> None:
        self.levels = 4
        self.members = 4
        self.channels = len(B2_FIELDS)
        self.modes = 88 // 2 + 1
        eligible = np.asarray(eligible_xy_mask, dtype=bool)
        if eligible.shape != (64, 32) or not np.any(eligible):
            raise ValueError("B4 stage spectral eligible mask differs")
        self.eligible = eligible
        self.field_absolute_error_sum = np.zeros(
            (self.levels, self.channels), dtype=np.float64
        )
        self.field_cell_count = 0
        self.truth_auto = np.zeros((self.channels, self.modes), dtype=np.float64)
        self.member_auto = np.zeros(
            (self.levels, self.members, self.channels, self.modes),
            dtype=np.float64,
        )
        self.mean_auto = np.zeros(
            (self.levels, self.channels, self.modes), dtype=np.float64
        )
        self.truth_mean_cross = np.zeros(
            (self.levels, self.channels, self.modes), dtype=np.complex128
        )
        self.pairs = tuple(
            (B2_FIELDS.index(first), B2_FIELDS.index(second))
            for first, second in B2_CROSS_PAIRS
        )
        self.truth_pair_cross = np.zeros(
            (len(self.pairs), self.modes), dtype=np.complex128
        )
        self.member_pair_cross = np.zeros(
            (self.levels, self.members, len(self.pairs), self.modes),
            dtype=np.complex128,
        )
        self.transport_truth = {
            quantity: [] for quantity in TRANSPORT_QUANTITIES
        }
        self.transport_members = {
            level: {quantity: [] for quantity in TRANSPORT_QUANTITIES}
            for level in range(self.levels)
        }
        self.targets = 0

    def update(
        self,
        *,
        standardized_stages: np.ndarray,
        standardized_truth: np.ndarray,
        physical_stages: np.ndarray,
        physical_truth: np.ndarray,
        native_truth: Mapping[str, np.ndarray],
        geometry: CodecTransportGeometry,
    ) -> None:
        standardized = np.asarray(standardized_stages, dtype=np.float64)
        standardized_target = np.asarray(standardized_truth, dtype=np.float64)
        physical = np.asarray(physical_stages, dtype=np.float64)
        physical_target = np.asarray(physical_truth, dtype=np.float64)
        if standardized.shape != (4, 4, 5, 64, 32, 88):
            raise ValueError("B4 standardized all-stage row shape differs")
        if standardized_target.shape != (5, 64, 32, 88):
            raise ValueError("B4 standardized stage truth shape differs")
        if physical.shape != (4, 4, 5, 64, 32, 88):
            raise ValueError("B4 physical all-stage row shape differs")
        if physical_target.shape != (5, 64, 32, 88):
            raise ValueError("B4 physical stage truth shape differs")
        if not all(
            np.all(np.isfinite(item))
            for item in (standardized, standardized_target, physical, physical_target)
        ):
            raise ValueError("B4 stage fields contain non-finite values")

        self.field_cell_count += int(np.prod(standardized_target.shape[1:]))
        truth_spectral = physical_target.copy()
        truth_spectral[3] -= np.mean(truth_spectral[3], dtype=np.float64)
        truth_z = np.fft.rfft(truth_spectral, axis=-1)[:, self.eligible, :]
        self.truth_auto += np.sum(
            np.abs(truth_z) ** 2, axis=1, dtype=np.float64
        )
        for pair_index, (first, second) in enumerate(self.pairs):
            self.truth_pair_cross[pair_index] += np.sum(
                truth_z[first] * np.conjugate(truth_z[second]),
                axis=0,
                dtype=np.complex128,
            )

        truth_transport_once: dict[str, float] | None = None
        for level in range(self.levels):
            standardized_members = standardized[:, level].copy()
            standardized_target_level = standardized_target.copy()
            standardized_members[:, 3], standardized_target_level[3] = (
                gauge_fix_phi_channel(
                    standardized_members[:, 3], standardized_target_level[3]
                )
            )
            ensemble_mean = np.mean(standardized_members, axis=0)
            self.field_absolute_error_sum[level] += np.sum(
                np.abs(ensemble_mean - standardized_target_level),
                axis=(1, 2, 3),
                dtype=np.float64,
            )

            physical_members = physical[:, level].copy()
            physical_target_level = physical_target.copy()
            physical_members[:, 3], physical_target_level[3] = (
                gauge_fix_phi_channel(
                    physical_members[:, 3], physical_target_level[3]
                )
            )
            member_z = np.fft.rfft(physical_members, axis=-1)[
                :, :, self.eligible, :
            ]
            self.member_auto[level] += np.sum(
                np.abs(member_z) ** 2, axis=2, dtype=np.float64
            )
            mean_z = np.mean(member_z, axis=0)
            self.mean_auto[level] += np.sum(
                np.abs(mean_z) ** 2, axis=1, dtype=np.float64
            )
            self.truth_mean_cross[level] += np.sum(
                truth_z * np.conjugate(mean_z),
                axis=1,
                dtype=np.complex128,
            )
            for pair_index, (first, second) in enumerate(self.pairs):
                self.member_pair_cross[level, :, pair_index] += np.sum(
                    member_z[:, first] * np.conjugate(member_z[:, second]),
                    axis=1,
                    dtype=np.complex128,
                )

            member_transport, truth_transport = _memberwise_separatrix_transport(
                physical_forecast_model88=physical[:, level],
                native_truth=native_truth,
                geometry=geometry,
            )
            if truth_transport_once is None:
                truth_transport_once = truth_transport
                for quantity in TRANSPORT_QUANTITIES:
                    self.transport_truth[quantity].append(
                        truth_transport[quantity]
                    )
            elif truth_transport != truth_transport_once:
                raise RuntimeError("B4 stage transport truth changed by level")
            for quantity in TRANSPORT_QUANTITIES:
                self.transport_members[level][quantity].append(
                    member_transport[quantity]
                )
        self.targets += 1

    def finalize(
        self,
        *,
        materiality: Mapping[str, Any],
        evaluate_gate: bool,
    ) -> dict[str, Any]:
        if self.targets <= 0 or self.field_cell_count <= 0:
            raise RuntimeError("B4 stage repair consumed no targets")
        weights = _one_sided_weights(88)
        expected_auto = np.mean(self.member_auto, axis=1)
        expected_cross = np.mean(self.member_pair_cross, axis=1)
        field_mae = self.field_absolute_error_sum / self.field_cell_count
        truth_mean_coherence = _safe_ratio(
            np.abs(self.truth_mean_cross) ** 2,
            self.truth_auto[None] * self.mean_auto,
        )

        truth_cross_coherence = np.empty(
            (len(self.pairs), self.modes), dtype=np.float64
        )
        forecast_cross_coherence = np.empty(
            (self.levels, len(self.pairs), self.modes), dtype=np.float64
        )
        for pair_index, (first, second) in enumerate(self.pairs):
            truth_cross_coherence[pair_index] = _safe_ratio(
                np.abs(self.truth_pair_cross[pair_index]) ** 2,
                self.truth_auto[first] * self.truth_auto[second],
            )
            forecast_cross_coherence[:, pair_index] = _safe_ratio(
                np.abs(expected_cross[:, pair_index]) ** 2,
                expected_auto[:, first] * expected_auto[:, second],
            )

        field_components: dict[int, list[dict[str, Any]]] = {
            level: [] for level in range(self.levels)
        }
        cross_components: dict[int, list[dict[str, Any]]] = {
            level: [] for level in range(self.levels)
        }
        for channel, field in enumerate(B2_FIELDS):
            for label, low, high in B2_MODE_BANDS:
                if not materiality["fields"][field]["bands"][label]["material"]:
                    continue
                indices = np.arange(low, high + 1, dtype=np.int64)
                truth_power = float(
                    np.sum(self.truth_auto[channel, indices] * weights[indices])
                )
                if not math.isfinite(truth_power) or truth_power <= 0.0:
                    raise FloatingPointError("B4 material truth field power is invalid")
                for level in range(self.levels):
                    predicted_power = float(
                        np.sum(
                            expected_auto[level, channel, indices]
                            * weights[indices]
                        )
                    )
                    ratio = predicted_power / truth_power
                    coherence = _weighted_mean(
                        truth_mean_coherence[level, channel, indices],
                        self.truth_auto[channel, indices] * weights[indices],
                    )
                    component = {
                        "field": field,
                        "band": label,
                        "stored_k": [low, high],
                        "full_torus_n": [5 * low, 5 * high],
                        "member_expected_power_ratio": ratio,
                        "absolute_log_power_ratio": abs(
                            math.log(max(ratio, 1.0e-12))
                        ),
                        "ensemble_mean_realization_coherence": coherence,
                        "one_minus_realization_coherence": 1.0 - coherence,
                    }
                    if not all(
                        math.isfinite(float(value))
                        for key, value in component.items()
                        if key
                        in {
                            "member_expected_power_ratio",
                            "absolute_log_power_ratio",
                            "ensemble_mean_realization_coherence",
                            "one_minus_realization_coherence",
                        }
                    ):
                        raise FloatingPointError("B4 material field component undefined")
                    field_components[level].append(component)

        for pair_index, pair in enumerate(B2_CROSS_PAIRS):
            pair_name = f"{pair[0]}-{pair[1]}"
            for label, low, high in B2_MODE_BANDS:
                if not materiality["cross_fields"][pair_name]["bands"][label][
                    "material"
                ]:
                    continue
                indices = np.arange(low, high + 1, dtype=np.int64)
                truth_complex = np.sum(self.truth_pair_cross[pair_index, indices])
                truth_amplitude = (
                    np.abs(self.truth_pair_cross[pair_index, indices])
                    * weights[indices]
                )
                if abs(truth_complex) == 0.0 or np.sum(truth_amplitude) <= 0.0:
                    raise FloatingPointError("B4 material truth cross spectrum invalid")
                for level in range(self.levels):
                    predicted_complex = np.sum(
                        expected_cross[level, pair_index, indices]
                    )
                    phase_radians = abs(
                        float(
                            np.angle(predicted_complex * np.conjugate(truth_complex))
                        )
                    )
                    coherence_error = _weighted_mean(
                        np.abs(
                            forecast_cross_coherence[level, pair_index, indices]
                            - truth_cross_coherence[pair_index, indices]
                        ),
                        truth_amplitude,
                    )
                    component = {
                        "pair": pair_name,
                        "band": label,
                        "stored_k": [low, high],
                        "full_torus_n": [5 * low, 5 * high],
                        "circular_phase_error_degrees": math.degrees(
                            phase_radians
                        ),
                        "circular_phase_error_over_pi": phase_radians / math.pi,
                        "truth_amplitude_weighted_absolute_coherence_error": (
                            coherence_error
                        ),
                        "combined_cross_error": phase_radians / math.pi
                        + coherence_error,
                    }
                    if not all(
                        math.isfinite(float(value))
                        for key, value in component.items()
                        if key
                        in {
                            "circular_phase_error_degrees",
                            "circular_phase_error_over_pi",
                            "truth_amplitude_weighted_absolute_coherence_error",
                            "combined_cross_error",
                        }
                    ):
                        raise FloatingPointError("B4 material cross component undefined")
                    cross_components[level].append(component)

        if not field_components[0] or not cross_components[0]:
            raise RuntimeError("B4 material stage component set is empty")
        level_records = []
        for level in range(self.levels):
            transport_components = []
            for quantity in TRANSPORT_QUANTITIES:
                truth = np.asarray(self.transport_truth[quantity], dtype=np.float64)
                members = np.asarray(
                    self.transport_members[level][quantity], dtype=np.float64
                )
                if truth.shape != (self.targets,) or members.shape != (
                    self.targets,
                    4,
                ):
                    raise RuntimeError("B4 stage transport accumulation shape differs")
                expected = np.mean(members, axis=1)
                denominator = float(np.sum(truth * truth, dtype=np.float64))
                relative_l2 = (
                    math.sqrt(
                        float(
                            np.sum((expected - truth) ** 2, dtype=np.float64)
                        )
                        / denominator
                    )
                    if denominator > 0.0
                    else math.nan
                )
                if not math.isfinite(relative_l2):
                    raise FloatingPointError("B4 stage transport relative L2 undefined")
                transport_components.append(
                    {
                        "quantity": quantity,
                        "separatrix_ensemble_expected_relative_L2": relative_l2,
                    }
                )
            record = {
                "level": level,
                "E_field": float(np.mean(field_mae[level])),
                "field_MAE_by_channel": {
                    field: float(field_mae[level, channel])
                    for channel, field in enumerate(B2_FIELDS)
                },
                "E_power": float(
                    np.mean(
                        [
                            item["absolute_log_power_ratio"]
                            for item in field_components[level]
                        ]
                    )
                ),
                "E_real": float(
                    np.mean(
                        [
                            item["one_minus_realization_coherence"]
                            for item in field_components[level]
                        ]
                    )
                ),
                "E_cross": float(
                    np.mean(
                        [item["combined_cross_error"] for item in cross_components[level]]
                    )
                ),
                "E_transport": float(
                    np.mean(
                        [
                            item["separatrix_ensemble_expected_relative_L2"]
                            for item in transport_components
                        ]
                    )
                ),
                "material_field_band_components": field_components[level],
                "material_cross_band_components": cross_components[level],
                "separatrix_transport_components": transport_components,
            }
            if not all(
                math.isfinite(record[name])
                for name in (
                    "E_field",
                    "E_power",
                    "E_real",
                    "E_cross",
                    "E_transport",
                )
            ):
                raise FloatingPointError("B4 stage aggregate error is undefined")
            level_records.append(record)

        initial = level_records[0]
        final = level_records[3]
        error_names = (
            "E_field",
            "E_power",
            "E_real",
            "E_cross",
            "E_transport",
        )
        if any(float(initial[name]) <= 0.0 for name in error_names):
            raise FloatingPointError(
                "B4 level-zero stage error denominator is nonpositive"
            )
        comparisons = {
            "E_field_final_over_level0": final["E_field"] / initial["E_field"],
            "E_power_final_over_level0": final["E_power"] / initial["E_power"],
            "E_real_final_over_level0": final["E_real"] / initial["E_real"],
            "E_cross_final_over_level0": final["E_cross"] / initial["E_cross"],
            "E_transport_final_over_level0": (
                final["E_transport"] / initial["E_transport"]
            ),
            "spectral_or_cross_strict_improvement": any(
                final[name] < initial[name]
                for name in ("E_power", "E_real", "E_cross")
            ),
            "transport_strict_improvement": (
                final["E_transport"] < initial["E_transport"]
            ),
        }
        if not all(
            math.isfinite(float(value))
            for key, value in comparisons.items()
            if key.endswith("_over_level0")
        ):
            raise FloatingPointError("B4 stage relative comparison is undefined")
        conditions = {
            "E_field_final_over_level0_at_most_1p05": (
                comparisons["E_field_final_over_level0"] <= 1.05
            ),
            "E_power_final_over_level0_at_most_1p05": (
                comparisons["E_power_final_over_level0"] <= 1.05
            ),
            "E_real_final_over_level0_at_most_1p05": (
                comparisons["E_real_final_over_level0"] <= 1.05
            ),
            "E_cross_final_over_level0_at_most_1p05": (
                comparisons["E_cross_final_over_level0"] <= 1.05
            ),
            "E_transport_final_over_level0_at_most_1p05": (
                comparisons["E_transport_final_over_level0"] <= 1.05
            ),
            "spectral_or_cross_strict_improvement": comparisons[
                "spectral_or_cross_strict_improvement"
            ],
            "transport_strict_improvement": comparisons[
                "transport_strict_improvement"
            ],
        }
        return {
            "target_count": self.targets,
            "members": 4,
            "levels": level_records,
            "aggregate_definitions": {
                "E_field": "equal_channel_standardized_MAE_of_M4_ensemble_mean",
                "E_power": "equal_material_field_band_mean_absolute_log_member_expected_power_ratio",
                "E_real": "equal_material_field_band_mean_one_minus_ensemble_mean_realization_coherence",
                "E_cross": "equal_material_cross_band_mean_phase_over_pi_plus_absolute_coherence_error",
                "E_transport": "equal_quantity_mean_relative_L2_of_memberwise_then_ensemble_expected_separatrix_transport",
            },
            "final_over_level0": comparisons,
            "gate_conditions": conditions if evaluate_gate else None,
            "gate_evaluated": bool(evaluate_gate),
            "passes": bool(all(conditions.values())) if evaluate_gate else None,
        }


def _score_stages(
    *,
    catalog: ModelDatasetCatalog,
    stage_artifact: Any,
    native_truth: NativeTruthCatalog,
    geometry: CodecTransportGeometry,
    event_threshold_record: Mapping[str, Any],
    target_frames: Sequence[int],
    bounded_smoke: bool,
) -> dict[str, Any]:
    sources = verify_locked_b4_metric_sources()
    _validate_artifact_identity(stage_artifact, stages=True)
    targets = strict_o2_targets(
        target_frames, split="validation", context_frames=1
    )
    required = (
        tuple(range(498, 502)) if bounded_smoke else tuple(range(498, 624))
    )
    if targets != required:
        purpose = "bounded smoke" if bounded_smoke else "scientific"
        raise ValueError(f"{purpose} B4 stage scoring target interval differs")
    if stage_artifact.target_frames != targets:
        raise ValueError("B4 stage artifact/scorer target frames differ")
    materiality = event_threshold_record.get("spectral_materiality", {})
    validate_b2_spectral_materiality(materiality)
    masks = geometry.region_masks
    accumulator = _StageRepairAccumulator(
        eligible_xy_mask=np.asarray(
            masks.strict_wall_interior & masks.operator_interior,
            dtype=bool,
        )
    )
    truth_dataset = OneStepWindowDataset(
        catalog,
        split="validation",
        target_frames=targets,
        context_frames=1,
        augment=False,
        seed=1701,
        return_physical=True,
    )
    try:
        for position, target in enumerate(targets):
            item = truth_dataset[position]
            if int(item["target_frame_index"]) != target:
                raise RuntimeError("B4 stage truth scoring order differs")
            standardized_stages = stage_artifact.read(position, position + 1)[0]
            if not all(
                np.array_equal(standardized_stages[0, 0], standardized_stages[m, 0])
                for m in range(1, 4)
            ):
                raise RuntimeError("B4 stored level zero is not member-shared")
            physical_stages = np.stack(
                [
                    decode_stage_member_forecasts(
                        catalog, standardized_stages[:, level]
                    )
                    for level in range(4)
                ],
                axis=1,
            )
            native_target = native_truth.read(
                target,
                target + 1,
                fields=("Ne", "Pe", "Pi", "phi"),
            )
            accumulator.update(
                standardized_stages=standardized_stages,
                standardized_truth=np.asarray(item["target"], dtype=np.float32),
                physical_stages=physical_stages,
                physical_truth=np.asarray(item["physical_target"], dtype=np.float64),
                native_truth=native_target,
                geometry=geometry,
            )
    finally:
        truth_dataset.close()
    repair = accumulator.finalize(
        materiality=materiality,
        evaluate_gate=not bounded_smoke,
    )
    return {
        "schema_version": 1,
        "scope": (
            "bounded_non_scientific_B4_PDE_Refiner_H1_stage_scoring_85604"
            if bounded_smoke
            else "B4_PDE_Refiner_H1_stagewise_H_det_repair_scoring_85604"
        ),
        "bounded_non_scientific_smoke": bool(bounded_smoke),
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_during_forecast_generation": False,
        "truth_opened_only_after_both_forecasts_closed_and_hash_verified": True,
        "training_performed": False,
        "physics_derived_training_loss_used": False,
        "model_seed": 1701,
        "target_frames": [targets[0], targets[-1] + 1],
        "target_count": len(targets),
        "stage_artifact": {
            "path": str(stage_artifact.path.resolve(strict=True)),
            "sha256": stage_artifact.sha256,
            "metadata": stage_artifact.metadata,
            "timing": stage_artifact.timing_record(),
        },
        "metric_sources": sources,
        "stagewise_repair": repair,
    }


def score_pde_refiner_stages(
    *,
    catalog: ModelDatasetCatalog,
    stage_artifact: Any,
    native_truth: NativeTruthCatalog,
    geometry: CodecTransportGeometry,
    event_threshold_record: Mapping[str, Any],
    target_frames: Sequence[int],
) -> dict[str, Any]:
    """Evaluate the full prospectively frozen four-level H-det repair test."""

    return _score_stages(
        catalog=catalog,
        stage_artifact=stage_artifact,
        native_truth=native_truth,
        geometry=geometry,
        event_threshold_record=event_threshold_record,
        target_frames=target_frames,
        bounded_smoke=False,
    )


def score_pde_refiner_stages_smoke(
    *,
    catalog: ModelDatasetCatalog,
    stage_artifact: Any,
    native_truth: NativeTruthCatalog,
    geometry: CodecTransportGeometry,
    event_threshold_record: Mapping[str, Any],
    target_frames: Sequence[int],
) -> dict[str, Any]:
    """Exercise identical stage mechanics on four non-scientific targets."""

    return _score_stages(
        catalog=catalog,
        stage_artifact=stage_artifact,
        native_truth=native_truth,
        geometry=geometry,
        event_threshold_record=event_threshold_record,
        target_frames=target_frames,
        bounded_smoke=True,
    )
