"""Tests for locked final scoring and the B4 stage-repair reduction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import tcv_diagnostics.pde_refiner_scoring as scoring
from tcv_diagnostics.b2_field_metrics import B2_FIELDS
from tcv_diagnostics.b2_spectral_metrics import B2_CROSS_PAIRS, B2_MODE_BANDS
from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES


def _artifact(**changes):
    metadata = {
        "source_kind": "selected_B4_PDE_Refiner",
        "arm": "B4-PDE-Refiner-H1",
        "seed": 1701,
        "context_frames": 1,
        "target_truth_read": False,
        "absolute_time_input": False,
        "member_prefixes_regenerated": False,
        "posthoc_calibration": False,
    }
    metadata.update(changes)
    return SimpleNamespace(model_seed=1701, metadata=metadata)


def _delegated(*, smoke: bool) -> dict:
    return {
        "schema_version": 1,
        "scope": (
            "bounded_non_scientific_B2_evaluator_smoke_scoring_85604"
            if smoke
            else "B2_truth_separated_probabilistic_scoring_85604"
        ),
        "development_run": "85604",
        "held_out_85606_read": False,
        "truth_opened_only_after_forecast_was_closed_and_hash_verified": True,
        "field_and_marginal_calibration": {"unchanged": 1.25},
    }


def _materiality() -> dict:
    return {
        "fields": {
            field: {
                "bands": {
                    label: {"material": field == "Ne" and label == "k1_3"}
                    for label, _, _ in B2_MODE_BANDS
                }
            }
            for field in B2_FIELDS
        },
        "cross_fields": {
            f"{first}-{second}": {
                "bands": {
                    label: {
                        "material": first == "Ne"
                        and second == "phi"
                        and label == "k1_3"
                    }
                    for label, _, _ in B2_MODE_BANDS
                }
            }
            for first, second in B2_CROSS_PAIRS
        },
    }


def test_locked_b4_metric_sources_match_frozen_protocol() -> None:
    assert scoring.verify_locked_b4_metric_sources() == scoring.LOCKED_METRIC_SOURCES


def test_final_wrapper_delegates_once_and_only_relabels_provenance(
    monkeypatch,
) -> None:
    delegated = _delegated(smoke=False)
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return delegated

    monkeypatch.setattr(scoring, "score_b2_forecast", fake)
    result = scoring.score_pde_refiner_final(
        catalog="catalog",
        forecast_artifact=_artifact(),
        native_truth="truth",
        geometry="geometry",
        event_threshold_record={"threshold": 1},
        target_frames=tuple(range(498, 624)),
    )
    assert len(calls) == 1
    assert calls[0]["model_seed"] == 1701
    assert result["scope"] == "B4_PDE_Refiner_H1_final_M32_scoring_85604"
    assert result["model_arm"] == "B4-PDE-Refiner-H1"
    assert result["context_frames"] == 1
    assert result["refinement_stage"] == 3
    assert result["field_and_marginal_calibration"] is delegated[
        "field_and_marginal_calibration"
    ]
    assert delegated["scope"] == "B2_truth_separated_probabilistic_scoring_85604"
    assert result["metric_engine"][
        "numerical_definitions_changed_for_B4_final"
    ] is False


def test_final_smoke_wrapper_uses_bounded_scorer(monkeypatch) -> None:
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return _delegated(smoke=True)

    monkeypatch.setattr(scoring, "score_b2_forecast_smoke", fake)
    result = scoring.score_pde_refiner_final_smoke(
        catalog=None,
        forecast_artifact=_artifact(),
        native_truth=None,
        geometry=None,
        event_threshold_record={},
        target_frames=tuple(range(498, 502)),
    )
    assert len(calls) == 1
    assert result["scope"].startswith(
        "bounded_non_scientific_B4_PDE_Refiner_H1"
    )


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("arm", "not-B4"),
        ("context_frames", 2),
        ("target_truth_read", True),
        ("absolute_time_input", True),
        ("member_prefixes_regenerated", True),
        ("posthoc_calibration", True),
    ),
)
def test_final_wrapper_rejects_wrong_b4_identity(key, value) -> None:
    with pytest.raises(ValueError, match="identity"):
        scoring.score_pde_refiner_final_smoke(
            catalog=None,
            forecast_artifact=_artifact(**{key: value}),
            native_truth=None,
            geometry=None,
            event_threshold_record={},
            target_frames=tuple(range(498, 502)),
        )


def test_stage_repair_reducer_uses_all_frozen_errors_and_reports_gate() -> None:
    accumulator = scoring._StageRepairAccumulator(  # noqa: SLF001
        eligible_xy_mask=np.ones((64, 32), dtype=bool)
    )
    accumulator.targets = 2
    accumulator.field_cell_count = 10
    for level in range(4):
        factor = 1.20 - 0.05 * level
        accumulator.field_absolute_error_sum[level] = factor
        accumulator.truth_auto[:] = 10.0
        accumulator.member_auto[level] = 10.0 * factor
        accumulator.mean_auto[level] = 10.0 * factor
        coherence = 0.80 + 0.04 * level
        accumulator.truth_mean_cross[level] = np.sqrt(
            coherence
            * accumulator.truth_auto
            * accumulator.mean_auto[level]
        )
        phase = 0.20 - 0.04 * level
        accumulator.truth_pair_cross[:] = 1.0 + 0.0j
        accumulator.member_pair_cross[level] = np.exp(1j * phase)
        for quantity in TRANSPORT_QUANTITIES:
            accumulator.transport_members[level][quantity] = [
                np.full(4, 1.0 + 0.10 * (4 - level)),
                np.full(4, 2.0 + 0.20 * (4 - level)),
            ]
    for quantity in TRANSPORT_QUANTITIES:
        accumulator.transport_truth[quantity] = [1.0, 2.0]

    result = accumulator.finalize(materiality=_materiality(), evaluate_gate=True)
    assert result["members"] == 4
    assert len(result["levels"]) == 4
    assert set(result["levels"][0]) >= {
        "E_field",
        "E_power",
        "E_real",
        "E_cross",
        "E_transport",
    }
    assert result["levels"][3]["E_field"] < result["levels"][0]["E_field"]
    assert result["levels"][3]["E_real"] < result["levels"][0]["E_real"]
    assert result["levels"][3]["E_cross"] < result["levels"][0]["E_cross"]
    assert result["levels"][3]["E_transport"] < result["levels"][0][
        "E_transport"
    ]
    assert result["gate_evaluated"] is True
    assert result["passes"] is True


def test_stage_decoder_rejects_noncanonical_shape_before_catalog_use() -> None:
    with pytest.raises(ValueError, match="shape"):
        scoring.decode_stage_member_forecasts(
            object(), np.zeros((4, 5, 2, 3, 4), dtype=np.float32)
        )
