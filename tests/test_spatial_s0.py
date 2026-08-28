from __future__ import annotations

import numpy as np
import pytest

from tcv_diagnostics.spatial_s0 import (
    FOOTPRINT_CENTERS,
    RIDGE_LAMBDAS,
    VOLUME_SHAPE,
    DualRidgeKernel,
    basic_metrics,
    build_fixed_footprints,
    choose_regularization,
    group_footprints,
    minimum_cylindrical_distance_to_observations,
    observe_density,
    select_median_hero_frame,
    toroidal_mode_metrics,
)


def test_fixed_footprints_wrap_only_toroidally() -> None:
    valid = np.ones(VOLUME_SHAPE[:2], dtype=bool)
    footprints, omitted = build_fixed_footprints(valid)
    assert omitted == ()
    grouped = group_footprints(footprints)
    assert {key: len(value) for key, value in grouped.items()} == {
        "A": 6,
        "B": 6,
        "C": 6,
    }
    first = grouped["A"][0]
    x, y, z = np.unravel_index(first.flat_indices, VOLUME_SHAPE)
    assert set(x) == {12, 13, 14}
    assert set(y) == {16, 17, 18}
    assert set(z) == {0, 1, 2, 86, 87}
    assert first.retained_cells == 45


def test_geometry_rule_omits_but_never_moves_failed_center() -> None:
    valid = np.ones(VOLUME_SHAPE[:2], dtype=bool)
    # Retain only one of nine x-y cells around preregistered A channel 0.
    for x in range(12, 15):
        for y in range(16, 19):
            valid[x, y] = False
    valid[13, 17] = True
    footprints, omitted = build_fixed_footprints(valid)
    failed = [item for item in omitted if item.family == "A" and item.channel == 0]
    assert len(failed) == 1
    assert failed[0].center == FOOTPRINT_CENTERS["A"][0]
    assert failed[0].retained_cells == 5
    assert not any(
        item.family == "A" and item.channel == 0 for item in footprints
    )


def test_observation_is_literal_boxcar_mean() -> None:
    valid = np.ones(VOLUME_SHAPE[:2], dtype=bool)
    footprints, _ = build_fixed_footprints(valid)
    first = group_footprints(footprints)["A"][0]
    density = np.arange(np.prod(VOLUME_SHAPE), dtype=np.float64).reshape(
        VOLUME_SHAPE
    )
    observed = observe_density(density, (first,))
    expected = np.mean(density.ravel()[first.flat_indices])
    assert observed.shape == (1,)
    assert observed[0] == expected


def test_equivalent_dual_predictor_matches_literal_dual() -> None:
    generator = np.random.default_rng(9182)
    fit_inputs = generator.normal(size=(31, 5))
    query_inputs = generator.normal(size=(7, 5))
    fit_targets = generator.normal(size=(31, 13))
    kernel = DualRidgeKernel.fit(fit_inputs, 1e-2)
    literal = kernel.predict(query_inputs, fit_targets)
    equivalent = kernel.predict_equivalent_dual(query_inputs, fit_targets)
    np.testing.assert_allclose(equivalent, literal, rtol=2e-11, atol=2e-11)


def test_regularization_selection_uses_complete_grid_and_smallest_tie() -> None:
    records = [
        {
            "regularization": value,
            "equal_field_full_state_rmse": 1.0 + abs(np.log10(value) + 2.0),
            "heldout_c_rmse": 1.0 + abs(np.log10(value) + 2.0),
        }
        for value in RIDGE_LAMBDAS
    ]
    winner, normalized = choose_regularization(records)
    assert winner == 1e-2
    assert len(normalized) == len(RIDGE_LAMBDAS)
    with pytest.raises(ValueError, match="complete"):
        choose_regularization(records[:-1])


def test_basic_metrics_and_mechanical_hero_selection() -> None:
    truth = np.asarray([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
    prediction = truth + np.asarray([[0.5, 0.5], [0.1, 0.1], [0.3, 0.3]])
    metrics = basic_metrics(truth, prediction)
    assert metrics["count"] == 6
    assert metrics["rmse"] > 0.0
    frame, scores = select_median_hero_frame((496, 497, 498), truth, prediction)
    assert frame == 498
    assert scores.shape == (3,)


def test_mode_mapping_and_known_single_mode() -> None:
    z = np.arange(VOLUME_SHAPE[2])
    wave = np.sin(2.0 * np.pi * 5 * z / VOLUME_SHAPE[2])
    truth = np.broadcast_to(wave, (2, 2, 1, VOLUME_SHAPE[2])).copy()
    prediction = 0.5 * truth
    mask = np.ones((2, 1), dtype=bool)
    records = toroidal_mode_metrics(truth, prediction, mask)
    record = next(item for item in records if item["stored_k"] == 5)
    assert record["physical_n"] == 25
    assert record["retained_power_ratio"] == pytest.approx(0.25)
    assert next(item for item in records if item["stored_k"] == 4)[
        "physical_n"
    ] == 20


def test_cylindrical_distance_is_zero_on_observed_cells_and_periodic() -> None:
    valid = np.ones(VOLUME_SHAPE[:2], dtype=bool)
    footprints, _ = build_fixed_footprints(valid)
    first = group_footprints(footprints)["A"][0]
    radius = np.ones(VOLUME_SHAPE[:2], dtype=np.float64)
    vertical = np.zeros_like(radius)
    distance = minimum_cylindrical_distance_to_observations(
        radius, vertical, (first,)
    )
    x, y, z = np.unravel_index(first.flat_indices, VOLUME_SHAPE)
    np.testing.assert_allclose(distance[x, y, z], 0.0, atol=2e-8)
    # The footprint wraps across z=87/0; both ends of the wedge are nearby.
    assert distance[13, 17, 87] == pytest.approx(0.0, abs=2e-8)
    assert distance[13, 17, 0] == pytest.approx(0.0, abs=2e-8)
