from __future__ import annotations

import numpy as np
import pytest

from tcv_diagnostics.b2_acceptance import (
    DeterministicFieldComparatorAccumulator,
    deterministic_transport_comparator,
)
from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES


def test_deterministic_field_comparator_uses_b2_phi_gauge_policy() -> None:
    mask = np.ones((64, 32, 88), dtype=bool)
    accumulator = DeterministicFieldComparatorAccumulator(
        target_frames=(498, 499),
        eligible_mask=mask,
        validation_blocks=((498,), (499,)),
    )
    truth = np.zeros((5, 64, 32, 88), dtype=np.float32)
    forecast = np.zeros_like(truth)
    forecast[0] = 2.0
    forecast[3] = 10.0
    truth[3] = 3.0
    for target in (498, 499):
        accumulator.update(
            target_frame=target,
            standardized_forecast=forecast,
            standardized_truth=truth,
        )
    result = accumulator.finalize()
    fields = result["overall"]["fields"]
    assert fields["Ne"]["mae"] == pytest.approx(2.0)
    assert fields["Ne"]["rmse"] == pytest.approx(2.0)
    assert fields["phi"]["mae"] == pytest.approx(0.0)
    assert fields["phi"]["rmse"] == pytest.approx(0.0)
    assert result["overall"][
        "aggregate_equal_channel_mae_standardized"
    ] == pytest.approx(0.4)
    assert len(result["chronological_blocks"]) == 2


def _paired_metrics() -> dict[str, object]:
    return {
        "point_count": 126,
        "reference_rms": 1.0,
        "candidate_rms": 1.0,
        "relative_l2": 0.1,
        "normalized_bias": 0.01,
        "rms_ratio": 1.0,
        "pearson_correlation": 0.9,
        "pearson_correlation_defined": True,
        "weighted_sign_disagreement": 0.01,
        "weighted_sign_disagreement_defined": True,
    }


def _transport_scope(start: int, stop: int) -> dict[str, object]:
    length = stop - start
    quantities = {
        quantity: {
            "strict_faces": {"metrics": _paired_metrics()},
            "separatrix": {"metrics": _paired_metrics()},
        }
        for quantity in TRANSPORT_QUANTITIES
    }
    truth = {
        quantity: [float(index + 1) for index in range(length)]
        for quantity in TRANSPORT_QUANTITIES
    }
    forecast = {
        quantity: [float(index + 2) for index in range(length)]
        for quantity in TRANSPORT_QUANTITIES
    }
    return {
        "frames": length,
        "comparisons": {"truth_vs_forecast": {"quantities": quantities}},
        "surface_series_normalized": {"truth": truth, "forecast": forecast},
    }


def test_deterministic_transport_comparator_extracts_frozen_absolute_error() -> None:
    blocks = [[498 + 21 * index, 519 + 21 * index] for index in range(6)]
    score = {
        "scope": "O2_truth_separated_forecast_scoring",
        "development_run": "85604",
        "held_out_85606_read": False,
        "target_truth_used_during_forecast_generation": False,
        "target_frames": [498, 624],
        "validation_blocks": blocks,
        "transport": {
            "overall": _transport_scope(498, 624),
            "blocks": [_transport_scope(start, stop) for start, stop in blocks],
        },
    }
    result = deterministic_transport_comparator(score)
    assert result["overall"]["target_frames"] == [498, 624]
    assert len(result["chronological_blocks"]) == 6
    for quantity in TRANSPORT_QUANTITIES:
        assert result["overall"]["quantities"][quantity][
            "separatrix_absolute_error"
        ] == pytest.approx(1.0)


def test_deterministic_transport_comparator_rejects_held_out_access() -> None:
    score = {
        "scope": "O2_truth_separated_forecast_scoring",
        "development_run": "85604",
        "held_out_85606_read": True,
        "target_truth_used_during_forecast_generation": False,
        "target_frames": [498, 624],
        "validation_blocks": [[498 + 21 * i, 519 + 21 * i] for i in range(6)],
    }
    with pytest.raises(ValueError, match="contract"):
        deterministic_transport_comparator(score)

