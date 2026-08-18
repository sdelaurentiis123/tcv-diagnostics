from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import tcv_diagnostics.b2_scoring as scoring
from tcv_diagnostics.b2_scoring import (
    compute_b2_transport_event_thresholds,
    validate_b2_transport_event_thresholds,
)
from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES


class _FakeNativeTruth:
    def read(self, start, stop, *, fields):
        assert fields == ("Ne", "Pe", "Pi", "phi")
        frames = np.arange(start, stop, dtype=np.float64)[:, None, None, None]
        shape = (stop - start, 2, 3, 4)
        return {
            "Ne": np.broadcast_to(frames + 1.0, shape).copy(),
            "Pe": np.broadcast_to(frames + 2.0, shape).copy(),
            "Pi": np.broadcast_to(frames + 3.0, shape).copy(),
            "phi": np.broadcast_to(frames + 4.0, shape).copy(),
        }


def test_training_transport_event_thresholds_use_only_frames_zero_to_431(monkeypatch):
    observed = []

    def fake_evaluate(state, geometry):
        del geometry
        frame_values = np.mean(state["Ne"], axis=(1, 2, 3))
        observed.extend(frame_values.tolist())
        return {
            quantity: {
                "separatrix_wedge": (index + 1.0) * frame_values,
            }
            for index, quantity in enumerate(TRANSPORT_QUANTITIES)
        }

    monkeypatch.setattr(scoring, "evaluate_transport_state", fake_evaluate)
    record = compute_b2_transport_event_thresholds(
        native_truth=_FakeNativeTruth(),
        geometry=SimpleNamespace(),
        chunk_frames=17,
    )
    assert observed == list(np.arange(1.0, 433.0))
    assert record["training_frames"] == [0, 432]
    assert record["validation_frames_read"] is False
    assert record["held_out_85606_read"] is False
    expected = float(np.quantile(np.arange(1.0, 433.0), 0.9, method="linear"))
    assert record["thresholds"]["particle"] == pytest.approx(expected)
    assert record["thresholds"]["total_internal_energy"] == pytest.approx(
        4.0 * expected
    )
    assert validate_b2_transport_event_thresholds(record) == record["thresholds"]


def test_transport_event_threshold_contract_rejects_validation_or_heldout_reads():
    record = {
        "scope": "B2_training_only_transport_event_thresholds",
        "development_run": "85604",
        "training_frames": [0, 432],
        "validation_frames_read": False,
        "held_out_85606_read": False,
        "quantile_probability": 0.90,
        "quantile_method": "numpy_linear",
        "absolute_value_before_quantile": True,
        "thresholds": {
            quantity: float(index + 1)
            for index, quantity in enumerate(TRANSPORT_QUANTITIES)
        },
        "physics_derived_training_loss_used": False,
    }
    assert validate_b2_transport_event_thresholds(record)["particle"] == 1.0
    contaminated = dict(record)
    contaminated["validation_frames_read"] = True
    with pytest.raises(ValueError, match="contract"):
        validate_b2_transport_event_thresholds(contaminated)
    contaminated = dict(record)
    contaminated["held_out_85606_read"] = True
    with pytest.raises(ValueError, match="contract"):
        validate_b2_transport_event_thresholds(contaminated)


def test_transport_event_threshold_contract_rejects_wrong_quantity_order():
    record = {
        "scope": "B2_training_only_transport_event_thresholds",
        "development_run": "85604",
        "training_frames": [0, 432],
        "validation_frames_read": False,
        "held_out_85606_read": False,
        "quantile_probability": 0.90,
        "quantile_method": "numpy_linear",
        "absolute_value_before_quantile": True,
        "thresholds": {
            quantity: 1.0 for quantity in reversed(TRANSPORT_QUANTITIES)
        },
        "physics_derived_training_loss_used": False,
    }
    with pytest.raises(ValueError, match="order"):
        validate_b2_transport_event_thresholds(record)
