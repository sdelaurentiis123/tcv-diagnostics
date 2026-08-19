"""B5 identity wrapper around the frozen B2 scientific metric engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .b2_scoring import score_b2_forecast, score_b2_forecast_smoke
from .codec_training import sha256_path


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


def verify_locked_metric_sources() -> dict[str, str]:
    """Fail closed if any pre-B5 numerical implementation changed."""

    verified: dict[str, str] = {}
    for relative, expected in LOCKED_METRIC_SOURCES.items():
        actual = sha256_path(ROOT / relative)
        if actual != expected:
            raise RuntimeError(
                f"frozen B5 metric source differs for {relative}: {actual}"
            )
        verified[relative] = actual
    return verified


def _validate_b5_artifact_identity(forecast_artifact: Any) -> None:
    metadata = forecast_artifact.metadata
    if (
        forecast_artifact.model_seed != 1701
        or metadata.get("source_kind") != "selected_B5_residual_EDM"
        or metadata.get("arm") != "B5-H1-JOINT-FIELD-EDM-UNET3D-MINI"
        or metadata.get("seed") != 1701
        or metadata.get("context_frames") != 1
        or metadata.get("target_truth_read") is not False
        or metadata.get("absolute_time_input") is not False
        or metadata.get("member_prefixes_regenerated") is not False
        or metadata.get("posthoc_calibration") is not False
    ):
        raise ValueError("B5 forecast artifact identity differs")


def _relabel(
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
        or score.get("truth_opened_only_after_forecast_was_closed_and_hash_verified")
        is not True
    ):
        raise RuntimeError("delegated frozen metric-engine result differs")
    result = dict(score)
    result["scope"] = (
        "bounded_non_scientific_B5_residual_EDM_evaluator_smoke_scoring_85604"
        if bounded_smoke
        else "B5_residual_EDM_truth_separated_probabilistic_scoring_85604"
    )
    result["model_arm"] = "B5-H1-JOINT-FIELD-EDM-UNET3D-MINI"
    result["context_frames"] = 1
    result["metric_engine"] = {
        "identity": "byte_locked_B2_numerical_metric_engine",
        "numerical_definitions_changed_for_B5": False,
        "original_delegated_scope": expected_scope,
        "source_sha256": dict(verified_sources),
    }
    return result


def _score_b5_forecast(
    *,
    catalog: Any,
    forecast_artifact: Any,
    native_truth: Any,
    geometry: Any,
    event_threshold_record: Mapping[str, Any],
    target_frames: Sequence[int],
    bounded_smoke: bool,
) -> dict[str, Any]:
    sources = verify_locked_metric_sources()
    _validate_b5_artifact_identity(forecast_artifact)
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
    return _relabel(
        score,
        bounded_smoke=bounded_smoke,
        verified_sources=sources,
    )


def score_b5_forecast(
    *,
    catalog: Any,
    forecast_artifact: Any,
    native_truth: Any,
    geometry: Any,
    event_threshold_record: Mapping[str, Any],
    target_frames: Sequence[int],
) -> dict[str, Any]:
    """Score the full B5 M32 artifact with unchanged B2 metrics."""

    return _score_b5_forecast(
        catalog=catalog,
        forecast_artifact=forecast_artifact,
        native_truth=native_truth,
        geometry=geometry,
        event_threshold_record=event_threshold_record,
        target_frames=target_frames,
        bounded_smoke=False,
    )


def score_b5_forecast_smoke(
    *,
    catalog: Any,
    forecast_artifact: Any,
    native_truth: Any,
    geometry: Any,
    event_threshold_record: Mapping[str, Any],
    target_frames: Sequence[int],
) -> dict[str, Any]:
    """Run the identical scorer on four targets as a non-scientific preflight."""

    return _score_b5_forecast(
        catalog=catalog,
        forecast_artifact=forecast_artifact,
        native_truth=native_truth,
        geometry=geometry,
        event_threshold_record=event_threshold_record,
        target_frames=target_frames,
        bounded_smoke=True,
    )
