"""Synthetic checks for the frozen ECRD scientific evaluator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tcv_diagnostics.b5_covariance_localization import B5_FINITE_MEMBER_FACTOR
from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES
from tcv_diagnostics.ecrd_scoring import (
    ECRD_COVARIANCE_SKETCH_PROBES,
    ECRD_EVALUATION_BLOCKS,
    ECRD_EVALUATION_TARGETS,
    SpatialTransportCovarianceSketch,
    exact_local_transport_from_b2_outputs,
)


def test_ecrd_evaluation_blocks_are_the_frozen_three_way_partition() -> None:
    assert ECRD_EVALUATION_BLOCKS == {
        "V00": tuple(range(498, 540)),
        "V01": tuple(range(540, 582)),
        "V02": tuple(range(582, 624)),
    }
    assert tuple(
        target for block in ECRD_EVALUATION_BLOCKS.values() for target in block
    ) == ECRD_EVALUATION_TARGETS


def test_spatial_transport_covariance_sketch_matches_its_frozen_formula() -> None:
    generator = np.random.default_rng(99)
    members = generator.normal(size=(32, 16, 81))
    truth = generator.normal(size=(16, 81))
    forecast = {name: members + index for index, name in enumerate(TRANSPORT_QUANTITIES)}
    observed = {name: truth + index for index, name in enumerate(TRANSPORT_QUANTITIES)}
    accumulator = SpatialTransportCovarianceSketch(
        quantities=TRANSPORT_QUANTITIES,
        rows=16,
        n_z=81,
    )
    accumulator.update(forecast=forecast, truth=observed)
    result = accumulator.finalize()
    assert result["probe_count"] == ECRD_COVARIANCE_SKETCH_PROBES
    flat = members.reshape(32, -1)
    mean = np.mean(flat, axis=0)
    anomaly = flat - mean
    probes = accumulator.probe_bank
    predictive = B5_FINITE_MEMBER_FACTOR * (
        anomaly.T @ (anomaly @ probes)
    ) / 31.0
    error = truth.reshape(-1) - mean
    realized = error[:, None] * (error @ probes)[None]
    expected = np.linalg.norm(predictive - realized) / np.linalg.norm(realized)
    for name in TRANSPORT_QUANTITIES:
        assert np.isclose(
            result["quantities"][name]["relative_frobenius_error_sketch"],
            expected,
        )


def test_exact_local_transport_is_selected_from_authoritative_strict_order() -> None:
    strict = np.ones((4, 5), dtype=bool)
    separatrix = np.zeros((4, 5), dtype=bool)
    separatrix.reshape(-1)[:16] = True
    geometry = SimpleNamespace(
        strict_face_mask=strict,
        separatrix_face_mask=separatrix,
    )
    generator = np.random.default_rng(7)
    forecast_outputs = {}
    truth_outputs = {}
    selector = separatrix[strict]
    for index, name in enumerate(TRANSPORT_QUANTITIES):
        forecast_strict = generator.normal(size=(32, 20, 81)) + index
        truth_strict = generator.normal(size=(20, 81)) + index
        forecast_outputs[name] = {
            "strict_face_contributions": forecast_strict.reshape(32, -1),
            "separatrix_wedge": np.sum(
                forecast_strict[:, selector], axis=(1, 2)
            ),
        }
        truth_outputs[name] = {
            "strict_face_contributions": truth_strict.reshape(-1),
            "separatrix_wedge": np.asarray(
                [np.sum(truth_strict[selector], dtype=np.float64)]
            ),
        }
    forecast, truth, closure = exact_local_transport_from_b2_outputs(
        forecast_outputs=forecast_outputs,
        truth_outputs=truth_outputs,
        geometry=geometry,
    )
    assert closure < 1.0e-14
    assert tuple(forecast) == TRANSPORT_QUANTITIES
    assert tuple(truth) == TRANSPORT_QUANTITIES
    assert all(values.shape == (32, 16, 81) for values in forecast.values())
    assert all(values.shape == (16, 81) for values in truth.values())
