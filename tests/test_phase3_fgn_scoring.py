"""Tests for the B3 identity wrapper around frozen scientific metrics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import tcv_diagnostics.fgn_scoring as scoring


def _artifact(**changes):
    metadata = {
        "source_kind": "selected_B3_FGN",
        "arm": "B3-FGN-H1",
        "seed": 1701,
        "context_frames": 1,
        "target_truth_read": False,
        "absolute_time_input": False,
        "member_prefixes_regenerated": False,
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


def test_locked_metric_sources_match_frozen_manifest_hashes() -> None:
    assert scoring.verify_locked_metric_sources() == scoring.LOCKED_METRIC_SOURCES


def test_full_wrapper_delegates_once_and_only_relabels_provenance(monkeypatch) -> None:
    delegated = _delegated(smoke=False)
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return delegated

    monkeypatch.setattr(scoring, "score_b2_forecast", fake)
    result = scoring.score_fgn_forecast(
        catalog="catalog",
        forecast_artifact=_artifact(),
        native_truth="truth",
        geometry="geometry",
        event_threshold_record={"threshold": 1},
        target_frames=tuple(range(498, 624)),
    )
    assert len(calls) == 1
    assert calls[0]["model_seed"] == 1701
    assert result["scope"] == (
        "B3_FGN_H1_truth_separated_probabilistic_scoring_85604"
    )
    assert result["model_arm"] == "B3-FGN-H1"
    assert result["context_frames"] == 1
    assert result["field_and_marginal_calibration"] is delegated[
        "field_and_marginal_calibration"
    ]
    assert delegated["scope"] == "B2_truth_separated_probabilistic_scoring_85604"
    assert result["metric_engine"]["numerical_definitions_changed_for_B3"] is False


def test_smoke_wrapper_uses_bounded_scorer(monkeypatch) -> None:
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return _delegated(smoke=True)

    monkeypatch.setattr(scoring, "score_b2_forecast_smoke", fake)
    result = scoring.score_fgn_forecast_smoke(
        catalog=None,
        forecast_artifact=_artifact(),
        native_truth=None,
        geometry=None,
        event_threshold_record={},
        target_frames=tuple(range(498, 502)),
    )
    assert len(calls) == 1
    assert result["scope"].startswith("bounded_non_scientific_B3_FGN_H1")


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("arm", "not-B3"),
        ("context_frames", 2),
        ("target_truth_read", True),
        ("absolute_time_input", True),
        ("member_prefixes_regenerated", True),
    ),
)
def test_wrapper_rejects_wrong_fgn_identity(key, value) -> None:
    with pytest.raises(ValueError, match="identity"):
        scoring.score_fgn_forecast_smoke(
            catalog=None,
            forecast_artifact=_artifact(**{key: value}),
            native_truth=None,
            geometry=None,
            event_threshold_record={},
            target_frames=tuple(range(498, 502)),
        )
