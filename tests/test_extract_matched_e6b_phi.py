"""Known-answer tests for assembled matched-E6B potential output."""

from __future__ import annotations

import numpy as np

from paper0.tools.extract_matched_e6b_phi import NATIVE_SHAPE, frame_metrics


def test_truth_replay_gate_is_scale_aware() -> None:
    truth = np.ones(NATIVE_SHAPE, dtype=np.float64)
    candidate = truth.copy()
    candidate[3, 4, 5] += 9.0e-10
    passing = frame_metrics(truth, candidate, atol=5.0e-10, rtol=5.0e-10)
    assert passing["passes"]
    candidate[3, 4, 5] += 2.0e-10
    failing = frame_metrics(truth, candidate, atol=5.0e-10, rtol=5.0e-10)
    assert not failing["passes"]


def test_truth_replay_zero_error_is_exact() -> None:
    truth = np.arange(np.prod(NATIVE_SHAPE), dtype=np.float64).reshape(NATIVE_SHAPE)
    record = frame_metrics(truth, truth.copy(), atol=0.0, rtol=0.0)
    assert record["passes"]
    assert record["relative_l2"] == 0.0
    assert record["rmse"] == 0.0
