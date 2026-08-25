from __future__ import annotations

import numpy as np
import pytest

from tcv_diagnostics.b2_transport_metrics import (
    B2_TRANSPORT_REDUCTIONS,
    B2TransportAccumulator,
)
from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES
from tcv_diagnostics.metrics import fair_crps


def _transport_case(target_index: int):
    generator = np.random.default_rng(900 + target_index)
    forecast = {}
    truth = {}
    for quantity_index, quantity in enumerate(TRANSPORT_QUANTITIES):
        scale = float(quantity_index + 1)
        strict_truth = scale * np.linspace(-1.5, 2.0, 13) + 0.1 * target_index
        separatrix_truth = np.asarray(
            [scale * (0.8 + 0.35 * target_index)], dtype=np.float64
        )
        strict_forecast = strict_truth[None] + generator.normal(
            scale=0.2 * scale, size=(32, strict_truth.size)
        )
        separatrix_forecast = separatrix_truth[0] + generator.normal(
            scale=0.1 * scale, size=32
        )
        forecast[quantity] = {
            "strict_face_contributions": strict_forecast,
            "separatrix_wedge": separatrix_forecast,
        }
        truth[quantity] = {
            "strict_face_contributions": strict_truth,
            "separatrix_wedge": separatrix_truth,
        }
    return forecast, truth


def _thresholds(value: float = 0.5) -> dict[str, float]:
    return {quantity: value for quantity in TRANSPORT_QUANTITIES}


def test_transport_accumulator_scores_memberwise_reductions_prefixes_and_events():
    scorer = B2TransportAccumulator(
        model_seed=1701,
        target_frames=(498, 499),
        event_thresholds=_thresholds(),
        detailed=True,
    )
    cases = []
    for index, target in enumerate((498, 499)):
        forecast, truth = _transport_case(index)
        cases.append((forecast, truth))
        scorer.update(
            target_frame=target,
            forecast_outputs=forecast,
            truth_outputs=truth,
        )
    result = scorer.finalize()

    assert result["target_count"] == 2
    assert result["nonlinear_operator_applied_per_member_before_reduction"]
    assert result["transport_of_ensemble_mean_fields_used"] is False
    assert result["complete_experimental_heat_flux_claimed"] is False
    assert len(result["per_target"]) == 2
    particle = result["quantities"]["particle"]
    strict = particle["reductions"]["strict_face_contributions"]
    separatrix = particle["reductions"]["separatrix_wedge"]
    assert strict["ensemble_expected_paired_metrics"]["relative_l2"] < 0.2
    assert separatrix["ensemble_expected_paired_metrics"][
        "pearson_correlation_defined"
    ]
    assert tuple(strict["member_prefix_sensitivity"]) == (
        "M4",
        "M8",
        "M16",
        "M32",
    )
    assert strict["pooled_distribution"]["truth_count"] == 26
    assert strict["pooled_distribution"]["member_count"] == 26 * 32
    assert particle["upper_decile_event_conditioned"]["defined"]
    assert particle["upper_decile_event_conditioned"]["validation_event_count"] == 2

    forecast_all = np.concatenate(
        [case[0]["particle"]["strict_face_contributions"] for case in cases],
        axis=1,
    )
    truth_all = np.concatenate(
        [case[1]["particle"]["strict_face_contributions"] for case in cases]
    )
    assert strict["ensemble_probabilistic_metrics"]["fair_crps"] == pytest.approx(
        np.mean(fair_crps(forecast_all, truth_all, member_axis=0))
    )


def test_transport_mirror_reuses_diagnostics_and_matches_standalone_block():
    thresholds = _thresholds()
    overall = B2TransportAccumulator(
        model_seed=1701,
        target_frames=(498, 499),
        event_thresholds=thresholds,
        detailed=True,
    )
    mirror = B2TransportAccumulator(
        model_seed=1701,
        target_frames=(498,),
        event_thresholds=thresholds,
        detailed=False,
    )
    standalone = B2TransportAccumulator(
        model_seed=1701,
        target_frames=(498,),
        event_thresholds=thresholds,
        detailed=False,
    )
    first_forecast, first_truth = _transport_case(0)
    second_forecast, second_truth = _transport_case(1)
    overall.update(
        target_frame=498,
        forecast_outputs=first_forecast,
        truth_outputs=first_truth,
        mirrors=(mirror,),
    )
    overall.update(
        target_frame=499,
        forecast_outputs=second_forecast,
        truth_outputs=second_truth,
    )
    standalone.update(
        target_frame=498,
        forecast_outputs=first_forecast,
        truth_outputs=first_truth,
    )
    assert mirror.finalize() == standalone.finalize()
    assert overall.finalize()["detailed"] is True


def test_transport_accumulator_rejects_reordered_target_and_reduction_schema():
    scorer = B2TransportAccumulator(
        model_seed=1701,
        target_frames=(498,),
        event_thresholds=_thresholds(),
        detailed=False,
    )
    forecast, truth = _transport_case(0)
    with pytest.raises(ValueError, match="differs"):
        scorer.update(
            target_frame=499,
            forecast_outputs=forecast,
            truth_outputs=truth,
        )
    malformed = {
        quantity: {
            "separatrix_wedge": forecast[quantity]["separatrix_wedge"],
            "strict_face_contributions": forecast[quantity][
                "strict_face_contributions"
            ],
        }
        for quantity in TRANSPORT_QUANTITIES
    }
    assert tuple(malformed["particle"]) != B2_TRANSPORT_REDUCTIONS
    with pytest.raises(ValueError, match="reductions"):
        scorer.update(
            target_frame=498,
            forecast_outputs=malformed,
            truth_outputs=truth,
        )


def test_transport_sparse_targets_require_explicit_authorization():
    with pytest.raises(ValueError, match="contiguous"):
        B2TransportAccumulator(
            model_seed=1701,
            target_frames=(498, 502),
            event_thresholds=_thresholds(),
            detailed=False,
        )
    scorer = B2TransportAccumulator(
        model_seed=1701,
        target_frames=(498, 502),
        event_thresholds=_thresholds(),
        detailed=False,
        allow_sparse_targets=True,
    )
    for index, target in enumerate((498, 502)):
        forecast, truth = _transport_case(index)
        scorer.update(
            target_frame=target,
            forecast_outputs=forecast,
            truth_outputs=truth,
        )
    result = scorer.finalize()
    assert result["target_frames"] == [498, 502]
    assert result["target_frames_are_explicit_indices"] is True
