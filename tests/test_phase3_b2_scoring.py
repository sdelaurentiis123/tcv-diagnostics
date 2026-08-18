from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import tcv_diagnostics.b2_scoring as scoring
from tcv_diagnostics.b2_scoring import (
    compute_b2_spectral_materiality,
    compute_b2_transport_event_thresholds,
    inherited_b2_spectral_materiality,
    score_b2_forecast_smoke,
    validate_b2_spectral_materiality,
    validate_b2_transport_event_thresholds,
)
from tcv_diagnostics.b2_field_metrics import B2_FIELDS
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


class _FakeTrainingFields:
    frames = tuple(range(432))
    fields = B2_FIELDS
    return_physical = True

    def __getitem__(self, index):
        z = 2.0 * np.pi * np.arange(16, dtype=np.float64) / 16
        mode = np.cos(2.0 * z + 0.01 * index)
        physical = np.stack(
            [
                np.broadcast_to((channel + 1.0) * mode, (2, 2, 16))
                for channel in range(5)
            ],
            axis=0,
        )
        return {"frame_index": index, "physical_volume": physical}


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


def test_training_spectral_materiality_uses_n_equals_5k_and_training_only() -> None:
    record = compute_b2_spectral_materiality(
        dataset=_FakeTrainingFields(),  # type: ignore[arg-type]
        eligible_xy_mask=np.ones((2, 2), dtype=bool),
    )
    validate_b2_spectral_materiality(record)
    assert record["training_frames"] == [0, 432]
    assert record["validation_frames_read"] is False
    for field in B2_FIELDS:
        bands = record["fields"][field]["bands"]
        assert bands["k1_3"]["full_torus_n"] == [5, 15]
        assert bands["k1_3"]["material"] is True
        assert bands["k1_3"][
            "fraction_of_training_nonaxisymmetric_power"
        ] == pytest.approx(1.0)
        assert bands["k4_5"]["material"] is False
        assert bands["k6_7"]["material"] is False
    for pair in ("Ne-phi", "Pe-phi", "Pi-phi"):
        assert record["cross_fields"][pair]["bands"]["k1_3"]["material"]

    contaminated = dict(record)
    contaminated["validation_frames_read"] = True
    with pytest.raises(ValueError, match="contract"):
        validate_b2_spectral_materiality(contaminated)


def test_b2_materiality_inherits_the_frozen_o2_training_decisions() -> None:
    bands = {
        label: {"truth_fraction": 0.02, "material": True}
        for label in ("k1_3", "k4_5", "k6_7")
    }
    o2 = {
        "scope": "O2_training_truth_materiality",
        "development_run": "85604",
        "training_frames": [0, 432],
        "held_out_85606_read": False,
        "validation_truth_used_to_select_bands": False,
        "materiality": {
            "source_split": "85604_training_[0,432)",
            "minimum_fraction": 0.01,
            "view": {
                "fields": list(B2_FIELDS),
                "cross_pairs": [["Ne", "phi"], ["Pe", "phi"], ["Pi", "phi"]],
            },
            "fields": {field: dict(bands) for field in B2_FIELDS},
            "cross_pairs": {
                pair: dict(bands) for pair in ("Ne-phi", "Pe-phi", "Pi-phi")
            },
        },
    }
    record = inherited_b2_spectral_materiality(o2)
    validate_b2_spectral_materiality(record)
    assert record["inherited_without_refitting"] is True
    assert record["fields"]["Ne"]["bands"]["k4_5"][
        "fraction_of_training_nonaxisymmetric_power"
    ] == pytest.approx(0.02)
    assert record["cross_fields"]["Ne-phi"]["bands"]["k6_7"]["full_torus_n"] == [30, 35]
    validate_b2_spectral_materiality(json.loads(json.dumps(record, sort_keys=True)))


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


def test_transport_event_threshold_contract_accepts_sorted_json_key_order():
    record = {
        "scope": "B2_training_only_transport_event_thresholds",
        "development_run": "85604",
        "training_frames": [0, 432],
        "validation_frames_read": False,
        "held_out_85606_read": False,
        "quantile_probability": 0.90,
        "quantile_method": "numpy_linear",
        "absolute_value_before_quantile": True,
        "thresholds": {quantity: 1.0 for quantity in TRANSPORT_QUANTITIES},
        "physics_derived_training_loss_used": False,
    }
    persisted = json.loads(json.dumps(record, sort_keys=True))
    assert tuple(persisted["thresholds"]) != TRANSPORT_QUANTITIES
    assert (
        tuple(validate_b2_transport_event_thresholds(persisted)) == TRANSPORT_QUANTITIES
    )


def test_transport_event_threshold_contract_rejects_wrong_quantity_keys():
    record = {
        "scope": "B2_training_only_transport_event_thresholds",
        "development_run": "85604",
        "training_frames": [0, 432],
        "validation_frames_read": False,
        "held_out_85606_read": False,
        "quantile_probability": 0.90,
        "quantile_method": "numpy_linear",
        "absolute_value_before_quantile": True,
        "thresholds": {quantity: 1.0 for quantity in TRANSPORT_QUANTITIES[:-1]},
        "physics_derived_training_loss_used": False,
    }
    record["thresholds"]["not_a_transport_quantity"] = 1.0
    with pytest.raises(ValueError, match="keys"):
        validate_b2_transport_event_thresholds(record)


def test_bounded_smoke_scorer_rejects_any_interval_other_than_four_targets():
    with pytest.raises(ValueError, match="bounded smoke"):
        score_b2_forecast_smoke(
            catalog=None,  # type: ignore[arg-type]
            forecast_artifact=None,  # type: ignore[arg-type]
            native_truth=None,  # type: ignore[arg-type]
            geometry=None,  # type: ignore[arg-type]
            event_threshold_record={},
            target_frames=tuple(range(498, 503)),
            model_seed=1701,
        )
