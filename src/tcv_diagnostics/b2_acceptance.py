"""Frozen comparator and acceptance primitives for Paper 0 B2.

This module contains no model inference and opens no simulation files by
itself.  It converts already verified deterministic forecasts/scores and B2
probabilistic scores into compact, auditable records.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .b2_field_metrics import B2_FIELDS, gauge_fix_phi_channel
from .codec_transport import TRANSPORT_QUANTITIES


B2_DETERMINISTIC_QUANTITY_MAP = {
    "particle": "particle",
    "electron_internal_energy": "electron_internal_energy",
    "ion_internal_energy": "ion_internal_energy",
    "total_internal_energy": "total_internal_energy",
}


class DeterministicFieldComparatorAccumulator:
    """Accumulate gauge-consistent deterministic field errors by time block."""

    def __init__(
        self,
        *,
        target_frames: Sequence[int],
        eligible_mask: np.ndarray,
        validation_blocks: Sequence[Sequence[int]],
    ) -> None:
        targets = tuple(int(item) for item in target_frames)
        blocks = tuple(tuple(int(item) for item in block) for block in validation_blocks)
        if not targets or targets != tuple(range(targets[0], targets[-1] + 1)):
            raise ValueError("deterministic comparator targets must be contiguous")
        if tuple(item for block in blocks for item in block) != targets:
            raise ValueError("deterministic comparator blocks must partition targets")
        mask = np.asarray(eligible_mask, dtype=bool)
        if mask.shape != (64, 32, 88) or not np.any(mask):
            raise ValueError("deterministic comparator eligible mask differs")
        self.target_frames = targets
        self.blocks = blocks
        self.block_for_target = {
            target: index for index, block in enumerate(blocks) for target in block
        }
        self.eligible = mask
        self.overall = self._empty()
        self.by_block = [self._empty() for _ in blocks]
        self.cursor = 0

    @staticmethod
    def _empty() -> dict[str, dict[str, float | int]]:
        return {
            field: {"count": 0, "absolute_error_sum": 0.0, "squared_error_sum": 0.0}
            for field in B2_FIELDS
        }

    @staticmethod
    def _add(
        destination: dict[str, dict[str, float | int]],
        field: str,
        error: np.ndarray,
    ) -> None:
        values = np.asarray(error, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("deterministic comparator error vector differs")
        destination[field]["count"] = int(destination[field]["count"]) + int(
            values.size
        )
        destination[field]["absolute_error_sum"] = float(
            destination[field]["absolute_error_sum"]
        ) + float(np.sum(np.abs(values), dtype=np.float64))
        destination[field]["squared_error_sum"] = float(
            destination[field]["squared_error_sum"]
        ) + float(np.sum(values * values, dtype=np.float64))

    def update(
        self,
        *,
        target_frame: int,
        standardized_forecast: np.ndarray,
        standardized_truth: np.ndarray,
    ) -> None:
        if self.cursor >= len(self.target_frames):
            raise ValueError("deterministic comparator received too many targets")
        expected = self.target_frames[self.cursor]
        if int(target_frame) != expected:
            raise ValueError(
                f"deterministic comparator target {target_frame} differs from {expected}"
            )
        forecast = np.asarray(standardized_forecast, dtype=np.float64)
        truth = np.asarray(standardized_truth, dtype=np.float64)
        expected_shape = (len(B2_FIELDS), 64, 32, 88)
        if forecast.shape != expected_shape or truth.shape != expected_shape:
            raise ValueError("deterministic comparator field shape differs")
        if not np.all(np.isfinite(forecast)) or not np.all(np.isfinite(truth)):
            raise ValueError("deterministic comparator fields are non-finite")
        forecast = forecast.copy()
        truth = truth.copy()
        forecast[3:4], truth[3] = gauge_fix_phi_channel(
            forecast[3:4], truth[3]
        )
        block = self.by_block[self.block_for_target[expected]]
        for channel, field in enumerate(B2_FIELDS):
            error = (forecast[channel] - truth[channel])[self.eligible]
            self._add(self.overall, field, error)
            self._add(block, field, error)
        self.cursor += 1

    @staticmethod
    def _finalize_scope(
        sufficient: Mapping[str, Mapping[str, float | int]],
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for field in B2_FIELDS:
            item = sufficient[field]
            count = int(item["count"])
            if count <= 0:
                raise ValueError("deterministic comparator scope is empty")
            absolute = float(item["absolute_error_sum"])
            squared = float(item["squared_error_sum"])
            if not math.isfinite(absolute) or not math.isfinite(squared):
                raise ValueError("deterministic comparator sufficient statistics differ")
            fields[field] = {
                "scalar_count": count,
                "mae": absolute / count,
                "rmse": math.sqrt(squared / count),
            }
        return {
            "fields": fields,
            "aggregate_equal_channel_mae_standardized": float(
                np.mean([fields[field]["mae"] for field in B2_FIELDS])
            ),
            "aggregate_equal_channel_rmse_standardized": math.sqrt(
                float(np.mean([fields[field]["rmse"] ** 2 for field in B2_FIELDS]))
            ),
        }

    def finalize(self) -> dict[str, Any]:
        if self.cursor != len(self.target_frames):
            raise RuntimeError("deterministic comparator did not receive every target")
        return {
            "scope": "gauge_consistent_deterministic_C5P_H2_field_comparator",
            "target_frames": [self.target_frames[0], self.target_frames[-1] + 1],
            "potential_policy": (
                "subtract_full_spatial_mean_separately_per_forecast_and_truth_target"
            ),
            "region": "eligible_union",
            "overall": self._finalize_scope(self.overall),
            "chronological_blocks": [
                {
                    "target_frames": [block[0], block[-1] + 1],
                    **self._finalize_scope(sufficient),
                }
                for block, sufficient in zip(self.blocks, self.by_block)
            ],
        }


def deterministic_transport_comparator(
    score: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract gauge-invariant transport comparators from one frozen O2 score."""

    if (
        score.get("scope") != "O2_truth_separated_forecast_scoring"
        or score.get("development_run") != "85604"
        or score.get("held_out_85606_read") is not False
        or score.get("target_truth_used_during_forecast_generation") is not False
        or score.get("target_frames") != [498, 624]
        or score.get("validation_blocks")
        != [[498 + 21 * index, 519 + 21 * index] for index in range(6)]
    ):
        raise ValueError("deterministic O2 score contract differs")
    scopes = [score["transport"]["overall"], *score["transport"]["blocks"]]
    frame_ranges = [score["target_frames"], *score["validation_blocks"]]
    records = []
    for scope, frame_range in zip(scopes, frame_ranges):
        comparisons = scope["comparisons"]["truth_vs_forecast"]["quantities"]
        series = scope["surface_series_normalized"]
        quantities = {}
        for b2_name, old_name in B2_DETERMINISTIC_QUANTITY_MAP.items():
            truth = np.asarray(series["truth"][old_name], dtype=np.float64)
            forecast = np.asarray(series["forecast"][old_name], dtype=np.float64)
            if (
                truth.ndim != 1
                or forecast.shape != truth.shape
                or truth.size == 0
                or not np.all(np.isfinite(truth))
                or not np.all(np.isfinite(forecast))
            ):
                raise ValueError("deterministic separatrix time series differs")
            quantities[b2_name] = {
                "strict_face_contributions": dict(
                    comparisons[old_name]["strict_faces"]["metrics"]
                ),
                "separatrix_wedge": dict(
                    comparisons[old_name]["separatrix"]["metrics"]
                ),
                "separatrix_absolute_error": float(
                    np.mean(np.abs(forecast - truth), dtype=np.float64)
                ),
            }
        if int(scope["frames"]) != int(frame_range[1]) - int(frame_range[0]):
            raise ValueError("deterministic transport scope frame count differs")
        records.append({"target_frames": list(frame_range), "quantities": quantities})
    if len(records) != 7 or tuple(records[0]["quantities"]) != TRANSPORT_QUANTITIES:
        raise ValueError("deterministic transport comparator needs overall plus six blocks")
    return {
        "scope": "frozen_gauge_invariant_deterministic_C5P_H2_transport_comparator",
        "overall": records[0],
        "chronological_blocks": records[1:],
    }
