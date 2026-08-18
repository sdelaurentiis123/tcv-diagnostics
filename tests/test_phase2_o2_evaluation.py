from __future__ import annotations

import copy

import numpy as np
import pytest

from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES
from tcv_diagnostics.o2_evaluation import (
    O2_FIELDS,
    O2FieldAccumulator,
    O2MetricAccumulator,
    assert_transport_summary_finite,
    build_o2_seed_gate,
    build_o2_transport_gate,
    o2_training_materiality,
    reference_skill_gate,
    spectral_cross_gate,
    validation_blocks,
)


def test_validation_blocks_are_exactly_six_chronological_groups():
    targets = tuple(range(498, 624))
    blocks = validation_blocks(targets)
    assert len(blocks) == 6
    assert all(len(block) == 21 for block in blocks)
    assert tuple(frame for block in blocks for frame in block) == targets
    with pytest.raises(ValueError, match="498..623"):
        validation_blocks(tuple(range(497, 623)))


def test_field_metrics_use_training_normalized_anomalies_and_equal_channels():
    truth = np.arange(20, dtype=np.float64).reshape(2, 5, 1, 1, 2) - 7.0
    forecast = truth + 1.0
    accumulator = O2FieldAccumulator.empty()
    accumulator.update(truth[:1], forecast[:1])
    second = O2FieldAccumulator.empty()
    second.update(truth[1:], forecast[1:])
    accumulator.merge(second)
    result = accumulator.finalize()

    assert result["aggregate_equal_channel_rmse_standardized"] == pytest.approx(1.0)
    assert result["aggregate_equal_channel_mae_standardized"] == pytest.approx(1.0)
    for index, field in enumerate(O2_FIELDS):
        values = result["field_metrics_standardized"][field]
        expected_truth = truth[:, index].reshape(-1)
        expected_forecast = forecast[:, index].reshape(-1)
        assert values["rmse"] == pytest.approx(1.0)
        assert values["mae"] == pytest.approx(1.0)
        assert values["bias"] == pytest.approx(1.0)
        assert values["anomaly_correlation"] == pytest.approx(
            np.dot(expected_truth, expected_forecast)
            / np.sqrt(
                np.dot(expected_truth, expected_truth)
                * np.dot(expected_forecast, expected_forecast)
            )
        )


def _exact_metric_record() -> dict:
    z = np.arange(8, dtype=np.float64)
    wave = 2.0 + np.cos(2.0 * np.pi * z / 8.0)
    physical = np.empty((2, 5, 64, 32, 8), dtype=np.float64)
    for channel in range(5):
        physical[:, channel] = (channel + 1.0) * wave
    standardized = physical.copy()
    latest = np.roll(physical, -1, axis=-1)
    metrics = O2MetricAccumulator(n_z=8)
    metrics.update(
        standardized_truth=standardized,
        standardized_forecast=standardized,
        physical_truth=physical,
        physical_forecast=physical,
        physical_latest_context=latest,
    )
    return metrics.finalize()


def test_exact_forecast_has_unit_spectral_transfer_and_zero_cross_error():
    result = _exact_metric_record()
    assert result["aggregate_equal_channel_rmse_standardized"] == 0.0
    spectra = result["spectral_and_cross_field"]
    for field in O2_FIELDS:
        band = spectra["field_band_summaries"][field]["k1_3"]
        assert band["power_ratio"] == pytest.approx(1.0)
        assert band["truth_power_weighted_transfer_coherence"] == pytest.approx(1.0)
    for pair in ("Ne-phi", "Pe-phi", "Pi-phi"):
        band = spectra["cross_field_band_summaries"][pair]["k1_3"]
        assert band[
            "truth_cross_amplitude_weighted_absolute_phase_error_degrees"
        ] == pytest.approx(0.0)
        assert band[
            "truth_cross_amplitude_weighted_absolute_coherence_change"
        ] == pytest.approx(0.0)
    lifetime = result["one_step_mode_lifetime"]
    assert lifetime["descriptive_only_not_used_for_O2_gate"] is True


def _field_record(rmse: float, mae: float | None = None) -> dict:
    value = float(rmse)
    return {
        "aggregate_equal_channel_rmse_standardized": value,
        "aggregate_equal_channel_mae_standardized": value if mae is None else float(mae),
        "field_metrics_standardized": {
            field: {"rmse": value, "mae": value, "bias": 0.0}
            for field in O2_FIELDS
        },
    }


def test_reference_skill_uses_best_metric_specific_reference_and_six_blocks():
    candidate = _field_record(0.5, 0.6)
    references = {
        "persistence": _field_record(1.0, 1.0),
        "spectral_ar1": _field_record(0.8, 0.9),
        "linear_extrapolation": _field_record(0.7, 0.55),
    }
    blocks = [candidate] * 6
    reference_blocks = {name: [record] * 6 for name, record in references.items()}
    gate = reference_skill_gate(
        candidate=candidate,
        candidate_blocks=blocks,
        references=references,
        reference_blocks=reference_blocks,
        applicable_references=(
            "persistence",
            "spectral_ar1",
            "linear_extrapolation",
        ),
    )
    aggregate = gate["aggregate_vs_best_applicable_reference"]
    assert aggregate["best_rmse_reference"] == "linear_extrapolation"
    assert aggregate["best_mae_reference"] == "linear_extrapolation"
    assert aggregate["beats_best_rmse"] is True
    assert aggregate["beats_best_mae"] is False
    assert gate["passes"] is False


def _transport_summary(*, good: bool = True) -> dict:
    metrics = {
        "relative_l2": 0.1 if good else 0.8,
        "normalized_bias": 0.05,
        "pearson_correlation": 0.95,
        "weighted_sign_disagreement": 0.02,
    }
    quantities = {
        quantity: {
            reduction: {"metrics": dict(metrics)}
            for reduction in ("strict_faces", "separatrix")
        }
        for quantity in TRANSPORT_QUANTITIES
    }
    return {
        "comparisons": {
            "truth_vs_forecast": {
                "quantities": quantities,
            }
        }
    }


def test_transport_gate_requires_overall_and_five_of_six_per_quantity_reduction():
    overall = _transport_summary()
    blocks = [_transport_summary() for _ in range(6)]
    assert_transport_summary_finite(overall)
    gate = build_o2_transport_gate(overall=overall, temporal_blocks=blocks)
    assert gate["passes"] is True

    one_bad = copy.deepcopy(blocks)
    one_bad[0] = _transport_summary(good=False)
    assert build_o2_transport_gate(
        overall=overall, temporal_blocks=one_bad
    )["passes"] is True

    two_bad = copy.deepcopy(one_bad)
    two_bad[1] = _transport_summary(good=False)
    assert build_o2_transport_gate(
        overall=overall, temporal_blocks=two_bad
    )["passes"] is False


def test_training_materiality_drives_spectral_and_cross_gate_without_validation_selection():
    exact = _exact_metric_record()["spectral_and_cross_field"]
    materiality = o2_training_materiality(exact)
    assert materiality["validation_truth_used_to_select_bands"] is False
    gate = spectral_cross_gate(
        candidate=exact,
        candidate_blocks=[exact] * 6,
        materiality=materiality,
    )
    assert gate["spectral"]["applicable_check_count"] > 0
    assert gate["cross_field"]["applicable_check_count"] > 0
    assert gate["passes"] is True


def test_complete_seed_gate_uses_h1_reference_applicability_and_all_physics_gates():
    candidate = _exact_metric_record()
    materiality = o2_training_materiality(
        candidate["spectral_and_cross_field"]
    )
    references = {}
    for name, error in (
        ("persistence", 1.0),
        ("spectral_ar1", 0.8),
        ("linear_extrapolation", 0.01),
    ):
        overall = {
            **_field_record(error),
            "spectral_and_cross_field": candidate["spectral_and_cross_field"],
        }
        references[name] = {
            "scientific_authority": True,
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "target_truth_used_during_forecast_generation": False,
            "target_frames": [498, 624],
            "target_count": 126,
            "field_spectral_cross": {
                "overall": overall,
                "blocks": [overall] * 6,
            },
        }
    candidate_score = {
        "scientific_authority": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_during_forecast_generation": False,
        "physics_derived_training_loss_used": False,
        "target_frames": [498, 624],
        "target_count": 126,
        "field_spectral_cross": {
            "overall": candidate,
            "blocks": [candidate] * 6,
        },
        "transport": {
            "overall": _transport_summary(),
            "blocks": [_transport_summary() for _ in range(6)],
        },
    }
    h1 = build_o2_seed_gate(
        arm="C5P-H1",
        candidate_score=candidate_score,
        reference_scores=references,
        materiality=materiality,
    )
    assert h1["applicable_references"] == ["persistence", "spectral_ar1"]
    assert h1["passes"] is True
    h2 = build_o2_seed_gate(
        arm="C5P-H2",
        candidate_score=candidate_score,
        reference_scores=references,
        materiality=materiality,
    )
    assert "linear_extrapolation" in h2["applicable_references"]
    assert h2["passes"] is True
