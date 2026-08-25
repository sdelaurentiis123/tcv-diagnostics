from __future__ import annotations

import numpy as np
import pytest

from tcv_diagnostics.b2_field_metrics import B2_ALL_REGIONS, B2_FIELDS
from tcv_diagnostics.b2_field_scoring import (
    B2_MEMBER_PREFIXES,
    B2FieldScoreAccumulator,
    PrefixFieldAccumulator,
)
from tcv_diagnostics.metrics import fair_crps


def _tiny_regions() -> dict[str, np.ndarray]:
    union = np.ones(12, dtype=bool)
    confined = np.zeros(12, dtype=bool)
    private = np.zeros(12, dtype=bool)
    sol = np.zeros(12, dtype=bool)
    confined[:4] = True
    private[4:8] = True
    sol[8:] = True
    masks = {
        "eligible_union": union,
        "confined_edge": confined,
        "private_flux": private,
        "scrape_off_layer": sol,
        "separatrix_cell_band": np.asarray(
            [
                False, False, True, True, True, True,
                False, False, False, False, False, False,
            ]
        ),
        "outboard_midplane": np.asarray(
            [
                True, False, False, True, False, False,
                True, False, False, True, False, False,
            ]
        ),
        "x_point_topology_stencil": np.asarray(
            [
                False, True, False, False, True, False,
                False, True, False, False, True, False,
            ]
        ),
        "inner_divertor_leg": np.asarray(
            [
                True, True, False, False, False, False,
                False, False, True, True, False, False,
            ]
        ),
        "outer_divertor_leg": np.asarray(
            [
                False, False, False, False, False, True,
                True, False, False, False, True, True,
            ]
        ),
    }
    assert tuple(masks) == B2_ALL_REGIONS
    return masks


def _tiny_case(seed: int = 9):
    generator = np.random.default_rng(seed)
    truth = generator.normal(size=(2, 5, 2, 2, 3))
    forecast = truth[:, None, None] + generator.normal(
        scale=0.4, size=(2, 32, 1, 5, 2, 2, 3)
    )
    return forecast, truth


def test_prefix_field_accumulator_matches_direct_score():
    generator = np.random.default_rng(3)
    forecast = generator.normal(size=(8, 15))
    truth = generator.normal(size=15)
    mask = np.asarray([True] * 10 + [False] * 5)
    accumulator = PrefixFieldAccumulator(8)
    accumulator.update_raw(forecast, truth, mask)
    record = accumulator.finalize()
    assert record["ensemble_size"] == 8
    assert record["fair_crps"] == pytest.approx(
        np.mean(fair_crps(forecast[:, mask], truth[mask], member_axis=0))
    )
    assert record["corrected_spread_skill_ratio"] > 0.0


def test_b2_field_score_streams_regions_blocks_prefixes_and_gauge_invariance():
    forecast, truth = _tiny_case()
    targets = (498, 499)
    blocks = ((498,), (499,))
    baseline = B2FieldScoreAccumulator(
        model_seed=1701,
        target_frames=targets,
        region_masks=_tiny_regions(),
        volume_shape=(2, 2, 3),
        validation_blocks=blocks,
    )
    shifted = B2FieldScoreAccumulator(
        model_seed=1701,
        target_frames=targets,
        region_masks=_tiny_regions(),
        volume_shape=(2, 2, 3),
        validation_blocks=blocks,
    )
    phi_member_offsets = np.linspace(-8.0, 13.0, 32)[None, :, None, None, None, None]
    for index, target in enumerate(targets):
        baseline.update(
            target_frame=target,
            standardized_forecast=forecast[index],
            standardized_truth=truth[index],
        )
        shifted_forecast = forecast[index].copy()
        shifted_truth = truth[index].copy()
        shifted_forecast[:, 0, 3] += phi_member_offsets[0, :, 0]
        shifted_truth[3] += 31.0
        shifted.update(
            target_frame=target,
            standardized_forecast=shifted_forecast,
            standardized_truth=shifted_truth,
        )
    first = baseline.finalize()
    second = shifted.finalize()

    assert first["target_count"] == 2
    assert first["fields"] == list(B2_FIELDS)
    assert tuple(first["regions"]) == B2_ALL_REGIONS
    assert len(first["chronological_blocks_eligible_union"]) == 2
    assert tuple(first["member_prefix_sensitivity_eligible_union"]) == tuple(
        f"M{members}" for members in B2_MEMBER_PREFIXES
    )
    assert first["conditional_uncertainty"] is None
    assert len(first["per_target_eligible_union_sufficient_statistics"]) == 2
    for region in B2_ALL_REGIONS:
        assert first["regions"][region]["aggregate"]["all_fields_nonzero_spread"]
        for metric in ("rmse", "mae", "bias"):
            assert first["regions"][region]["fields"]["phi"]["ensemble_mean"][
                metric
            ] == pytest.approx(
                second["regions"][region]["fields"]["phi"]["ensemble_mean"][
                    metric
                ],
                abs=2e-14,
            )
        assert first["regions"][region]["fields"]["phi"][
            "fair_crps"
        ] == pytest.approx(
            second["regions"][region]["fields"]["phi"]["fair_crps"],
            abs=2e-14,
        )
    assert second["raw_stored_gauge_phi_eligible_union_descriptive_only"][
        "ensemble_mean"
    ]["rmse"] > first["raw_stored_gauge_phi_eligible_union_descriptive_only"][
        "ensemble_mean"
    ]["rmse"]


def test_b2_field_score_rejects_target_reordering_and_incomplete_finalize():
    forecast, truth = _tiny_case()
    scorer = B2FieldScoreAccumulator(
        model_seed=1701,
        target_frames=(498, 499),
        region_masks=_tiny_regions(),
        volume_shape=(2, 2, 3),
    )
    with pytest.raises(ValueError, match="differs"):
        scorer.update(
            target_frame=499,
            standardized_forecast=forecast[0],
            standardized_truth=truth[0],
        )
    scorer.update(
        target_frame=498,
        standardized_forecast=forecast[0],
        standardized_truth=truth[0],
    )
    with pytest.raises(RuntimeError, match="every target"):
        scorer.finalize()
