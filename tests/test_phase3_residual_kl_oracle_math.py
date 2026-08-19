"""Known-answer tests for the residual-KL oracle mathematics."""

from __future__ import annotations

import numpy as np
import pytest

from tcv_diagnostics.residual_kl_oracle import (
    KL_FIELDS,
    KL_MASTER_SEED,
    center_training_residual,
    classify_kl_outcome,
    diagonalize_snapshot_gram,
    fit_snapshot_kl,
    gauge_fixed_residual,
    generate_seed_bank,
    project_onto_kl,
    reconstruct_static_kl_members,
    reconstruction_variance_capture,
    select_static_rank,
    snapshot_mode_block,
    static_standard_normal_coefficients,
    streaming_snapshot_gram,
)


def _fields(samples: int = 6, *, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(samples, len(KL_FIELDS), 2, 3, 4))


def test_phi_gauge_offsets_cancel_before_residual_formation() -> None:
    truth = _fields()
    mean = _fields(seed=12)
    expected = gauge_fixed_residual(truth, mean)
    truth[:, 3] += np.arange(truth.shape[0])[:, None, None, None] + 19.0
    mean[:, 3] -= 3.0 * np.arange(mean.shape[0])[:, None, None, None] - 7.0
    actual = gauge_fixed_residual(truth, mean)
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    assert np.max(np.abs(np.mean(actual[:, 3], axis=(1, 2, 3)))) < 2e-14


def test_bias_and_covariance_centering_are_separate() -> None:
    residual = _fields(samples=7)
    product = center_training_residual(residual)
    np.testing.assert_allclose(
        product.axisymmetric_bias,
        np.mean(residual, axis=(0, 4)),
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        np.mean(product.covariance_empirical_mean, axis=-1),
        0.0,
        atol=2e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.sum(product.covariance_centered, axis=0),
        0.0,
        atol=3e-14,
        rtol=0.0,
    )
    assert np.linalg.norm(product.covariance_empirical_mean) > 0.0
    assert product.maximum_relative_row_sum < 5e-13
    assert product.maximum_relative_empirical_mean_toroidal_average < 5e-13


def test_method_of_snapshots_recovers_known_orthonormal_subspace() -> None:
    # Four centered samples span two orthogonal feature directions.
    samples = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    basis = fit_snapshot_kl(samples)
    assert basis.positive_rank == 2
    np.testing.assert_allclose(basis.eigenvalues, [8.0 / 3.0, 2.0 / 3.0])
    np.testing.assert_allclose(
        basis.modes.reshape(2, -1) @ basis.modes.reshape(2, -1).T,
        np.eye(2),
        atol=2e-14,
    )
    assert basis.maximum_orthonormality_error < 2e-14
    assert basis.full_rank_training_relative_rms < 2e-14


def test_streaming_snapshot_blocks_match_in_memory_fit() -> None:
    centered = center_training_residual(_fields(samples=9)).covariance_centered
    flat = np.asarray(centered.reshape(centered.shape[0], -1), dtype=np.float32)
    gram, centering_error = streaming_snapshot_gram(
        flat,
        feature_chunk_size=17,
    )
    eigensystem = diagonalize_snapshot_gram(
        gram,
        sample_count=flat.shape[0],
        feature_count=flat.shape[1],
        maximum_relative_feature_mean=centering_error,
    )
    streamed_modes = np.concatenate(
        [
            snapshot_mode_block(flat[:, start : start + 17], eigensystem)
            for start in range(0, flat.shape[1], 17)
        ],
        axis=1,
    )
    direct = fit_snapshot_kl(centered)
    np.testing.assert_allclose(eigensystem.eigenvalues, direct.eigenvalues, rtol=2e-7)
    # Eigenvector signs are arbitrary, so compare the subspace projectors.
    np.testing.assert_allclose(
        streamed_modes.T @ streamed_modes,
        direct.modes.reshape(direct.positive_rank, -1).T
        @ direct.modes.reshape(direct.positive_rank, -1),
        atol=4e-7,
        rtol=4e-7,
    )


def test_projection_is_nested_and_full_rank_closes() -> None:
    residual = center_training_residual(_fields(samples=9)).covariance_centered
    basis = fit_snapshot_kl(residual)
    errors = []
    for rank in range(basis.positive_rank + 1):
        reconstruction = project_onto_kl(residual, basis, rank=rank)
        errors.append(float(np.linalg.norm(residual - reconstruction)))
    assert np.all(np.diff(errors) <= 2e-12)
    assert errors[-1] / np.linalg.norm(residual) < 2e-13


def test_variance_capture_is_total_and_fieldwise() -> None:
    reference = _fields(samples=4)
    perfect = reconstruction_variance_capture(reference, reference)
    assert perfect["total"] == pytest.approx(1.0)
    assert set(perfect["fields"]) == set(KL_FIELDS)
    assert all(value == pytest.approx(1.0) for value in perfect["fields"].values())
    zero = reconstruction_variance_capture(reference, np.zeros_like(reference))
    assert zero["total"] == pytest.approx(0.0)


def test_static_rank_selection_uses_only_training_eigenvalues() -> None:
    values = np.geomspace(1.0, 1e-5, 128)
    selected = select_static_rank(values)
    assert selected["rank"] in (8, 16, 32, 44, 64, 128)
    assert selected["validation_used"] is False
    cumulative = np.cumsum(values) / np.sum(values)
    if selected["label"] == "training_90_percent_rank":
        assert cumulative[selected["rank"] - 1] >= 0.90
        earlier = [rank for rank in (8, 16, 32, 44, 64, 128) if rank < selected["rank"]]
        assert all(cumulative[rank - 1] < 0.90 for rank in earlier)


def test_static_rank_cap_bound_is_not_silently_clipped() -> None:
    selected = select_static_rank(np.ones(200))
    assert selected["rank"] == 128
    assert selected["label"] == "training_variance_cap_bound"
    with pytest.raises(RuntimeError, match="rank 128 is unavailable"):
        select_static_rank(np.ones(100))


def test_seed_bank_and_coefficients_reload_exactly() -> None:
    first = generate_seed_bank(target_count=3, ensemble_size=4)
    second = generate_seed_bank(target_count=3, ensemble_size=4)
    assert first.dtype == np.uint64
    assert np.array_equal(first, second)
    assert np.unique(first).size == first.size
    coefficients = static_standard_normal_coefficients(first[0], rank=5)
    repeated = static_standard_normal_coefficients(first[0], rank=5)
    assert np.array_equal(coefficients, repeated)
    with pytest.raises(ValueError, match="master seed"):
        generate_seed_bank(master_seed=KL_MASTER_SEED + 1)


def test_static_reconstruction_has_frozen_mean_and_memberwise_anomalies() -> None:
    samples = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    basis = fit_snapshot_kl(samples)
    mean = np.asarray([10.0, 20.0, 30.0])
    bias = np.asarray(1.0)
    seeds = generate_seed_bank(target_count=1, ensemble_size=32)[0]
    members = reconstruct_static_kl_members(
        h1_mean=mean,
        axisymmetric_bias=bias,
        basis=basis,
        rank=2,
        member_seeds=seeds,
    )
    expected = static_standard_normal_coefficients(seeds, rank=2)
    expected = (
        expected * np.sqrt(basis.eigenvalues)[None]
    ) @ basis.modes.reshape(2, -1)
    np.testing.assert_allclose(
        members - (mean + bias)[None], expected, atol=4e-15, rtol=4e-15
    )
    # The third feature has no covariance support and therefore remains fixed.
    np.testing.assert_array_equal(members[:, 2], np.full(32, 31.0))


@pytest.mark.parametrize(
    ("rank", "useful", "expected"),
    [
        (32, True, "K1_compact_representation_static_covariance_useful"),
        (64, False, "K2_compact_representation_conditional_coefficients_required"),
        (128, False, "K3_only_moderate_or_high_rank_adequate"),
        (96, False, "inconsistent_diagnostic_requires_review"),
    ],
)
def test_outcome_classification(rank: int, useful: bool, expected: str) -> None:
    assert classify_kl_outcome(
        minimum_passing_rank=rank,
        full_positive_rank=200,
        tier_b_useful=useful,
    ) == expected
    assert classify_kl_outcome(
        minimum_passing_rank=None,
        full_positive_rank=200,
        tier_b_useful=False,
    ) == "K4_training_residual_span_does_not_transfer"
