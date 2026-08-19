from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tcv_diagnostics.phase3_5.context_shuffle import (
    SELECTED_TARGETS,
    mismatched_target,
)
from tcv_diagnostics.phase3_5.probes import (
    causal_context_features,
    evaluate_chronological_probes_multi,
    fit_shallow_tree,
    nearest_preceding_neighbors,
    predict_tree,
)
from tcv_diagnostics.phase3_5.representations import (
    FourierSeparatedRepresentation,
    HaarSubbandRepresentation,
    PatchwisePCARepresentation,
    assert_storage_not_above_global,
    haar_decompose,
    haar_reconstruct,
)
from tcv_diagnostics.phase3_5.scope import (
    load_phase3_5_protocol,
    validate_phase3_5_frames,
)
from tcv_diagnostics.phase3_5.statistics import (
    effective_sample_record,
    fit_snapshot_subspace_from_raw_gram,
    moving_block_indices,
    permute_complete_blocks,
    raw_sample_gram,
    snapshot_principal_angles,
    snapshot_transfer_capture,
)
from tcv_diagnostics.phase3_5.translation import (
    circular_toroidal_roll,
    estimate_toroidal_displacement,
    normalized_equivariance_error,
    training_field_rms,
)


ROOT = Path(__file__).resolve().parents[1]


def test_scope_loader_verifies_ranges_and_both_protocol_hashes() -> None:
    protocol = load_phase3_5_protocol(
        ROOT / "paper0/manifests/phase3_5_cause_localization_85604.json",
        root=ROOT,
    )
    assert len(protocol.training_blocks) == 10
    assert len(protocol.validation_blocks) == 3
    assert protocol.matched_sample_count == 42
    assert validate_phase3_5_frames(range(2, 432), split="train", targets=True)[0] == 2
    with pytest.raises(ValueError, match="guard|interval"):
        validate_phase3_5_frames(range(431, 433), split="train", targets=True)


def test_toroidal_shift_known_answer_and_nonperiodic_rejection() -> None:
    generator = np.random.default_rng(41)
    training = generator.normal(size=(12, 5, 4, 3, 88))
    field_rms = training_field_rms(training)
    first = training[0]
    later = np.roll(first, 11, axis=-1)
    estimate = estimate_toroidal_displacement(first, later, field_rms=field_rms)
    assert estimate.shared["signed_integer_shift"] == 11
    assert estimate.shared["peak_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert estimate.shared["subcell_shift"] == pytest.approx(11.0, abs=1e-8)
    assert np.array_equal(circular_toroidal_roll(first, 11), later)
    with pytest.raises(ValueError, match="only"):
        circular_toroidal_roll(first, 1, axis=-2)


def test_fourier_subcell_recovers_fractional_phase_ramp() -> None:
    generator = np.random.default_rng(17)
    first = generator.normal(size=(5, 3, 2, 88))
    spectrum = np.fft.rfft(first, axis=-1)
    shift = 9.35
    k = np.arange(spectrum.shape[-1])
    later = np.fft.irfft(
        spectrum * np.exp(-2j * np.pi * k * shift / 88), n=88, axis=-1
    )
    field_rms = np.sqrt(np.mean((first - first.mean(axis=-1, keepdims=True)) ** 2, axis=(1, 2, 3)))
    estimate = estimate_toroidal_displacement(first, later, field_rms=field_rms)
    assert estimate.shared["signed_integer_shift"] == 9
    assert estimate.shared["available"] is True
    assert estimate.shared["subcell_shift"] == pytest.approx(shift, abs=0.05)


def test_equivariance_error_formula_known_answer() -> None:
    reference = np.ones((2, 3, 4))
    assert normalized_equivariance_error(reference, reference) == 0.0
    assert normalized_equivariance_error(reference, 2 * reference) == pytest.approx(1.0)


def test_moving_block_bootstrap_is_noncircular_and_complete() -> None:
    indices = moving_block_indices(42, block_length=12, replicates=20, seed=5)
    assert indices.shape == (20, 42)
    assert np.all((indices >= 0) & (indices < 42))
    # Every run of 12 within the generated sequence is increasing until the
    # next independently drawn block; no modulo wrap is possible.
    for row in indices:
        for start in (0, 12, 24):
            assert np.all(np.diff(row[start : start + 12]) == 1)


def test_ess_known_white_noise_and_ar1_ordering() -> None:
    generator = np.random.default_rng(8)
    white = generator.normal(size=4000)
    ar = np.empty(4000)
    ar[0] = white[0]
    for index in range(1, ar.size):
        ar[index] = 0.9 * ar[index - 1] + white[index]
    white_record = effective_sample_record(white, detrend=False)
    ar_record = effective_sample_record(ar, detrend=False)
    assert white_record["primary_effective_sample_size"] > 2000
    assert ar_record["primary_effective_sample_size"] < 600
    assert ar_record["primary_tau_int"] > white_record["primary_tau_int"]


def test_snapshot_transfer_and_principal_angles_known_subspace() -> None:
    generator = np.random.default_rng(9)
    source = generator.normal(size=(42, 30))
    target = source.copy()
    values = np.concatenate((source, target))
    gram = raw_sample_gram(values)
    first = fit_snapshot_subspace_from_raw_gram(gram, range(42))
    second = fit_snapshot_subspace_from_raw_gram(gram, range(42, 84))
    capture = snapshot_transfer_capture(gram, first, range(42, 84), rank=30)
    angles = snapshot_principal_angles(gram, first, second, rank=20)
    assert capture == pytest.approx(1.0, abs=1e-10)
    assert angles["minimum_cosine"] == pytest.approx(1.0, abs=1e-8)


def test_haar_roundtrip_has_22_subbands() -> None:
    generator = np.random.default_rng(2)
    values = generator.normal(size=(3, 2, 8, 8, 8))
    parts = haar_decompose(values, levels=3)
    reconstruction = haar_reconstruct(parts, levels=3)
    assert len(parts) == 22
    assert np.max(np.abs(reconstruction - values)) < 1e-12


def test_fourier_complex_pairing_and_budget_accounting() -> None:
    generator = np.random.default_rng(12)
    values = generator.normal(size=(8, 2, 8, 8, 8))
    representation = FourierSeparatedRepresentation.fit(values)
    accounting = representation.accounting(5)
    assert accounting["real_coefficients"] <= 5
    assert all(component.real_cost in (1, 2) for component in representation.components)
    assert representation.components[0].real_cost == 1
    assert representation.components[-1].real_cost == 1
    assert_storage_not_above_global(accounting, global_budget=5, sample_shape=values.shape[1:])


def test_haar_and_patch_representations_reconstruct_shapes_and_cover_boundaries() -> None:
    generator = np.random.default_rng(19)
    values = generator.normal(size=(8, 2, 8, 8, 8))
    haar = HaarSubbandRepresentation.fit(values, levels=3)
    patch = PatchwisePCARepresentation.fit(
        values, patch_shape=(4, 4, 8), stride=(2, 2, 8)
    )
    assert haar.reconstruct(values, budget=6).shape == values.shape
    reconstructed = patch.reconstruct(values, budget=6)
    assert reconstructed.shape == values.shape
    assert np.all(np.isfinite(reconstructed))


def test_chronological_probe_features_do_not_accept_target_axis() -> None:
    generator = np.random.default_rng(23)
    context = generator.normal(size=(20, 5, 8, 4, 8))
    features, names = causal_context_features(context, radial_bins=4)
    assert features.shape[0] == context.shape[0]
    assert features.shape[1] == len(names)
    with pytest.raises(ValueError):
        causal_context_features(context[:, None], radial_bins=4)


def test_depth_two_tree_and_complete_block_permutations() -> None:
    x = np.linspace(-1, 1, 100)[:, None]
    y = (x[:, 0] > 0).astype(float)
    tree = fit_shallow_tree(x, y, maximum_depth=2, minimum_leaf=24)
    prediction = predict_tree(tree, x)
    assert np.mean((prediction - y) ** 2) < 0.05
    permutations = permute_complete_blocks((42,) * 10, replicates=10, seed=7)
    assert permutations.shape == (10, 420)
    for row in permutations:
        for start in range(0, 420, 42):
            assert np.all(np.diff(row[start : start + 42]) == 1)


def test_multi_target_chronological_probe_reports_every_target_and_block() -> None:
    generator = np.random.default_rng(27)
    train_x = generator.normal(size=(252, 7))
    validation_x = generator.normal(size=(84, 7))
    slopes = generator.normal(size=(7, 2))
    train_y = train_x @ slopes + 0.01 * generator.normal(size=(252, 2))
    validation_y = validation_x @ slopes + 0.01 * generator.normal(size=(84, 2))
    rows = evaluate_chronological_probes_multi(
        train_x,
        np.arange(252),
        train_y,
        validation_x,
        np.arange(252, 336),
        validation_y,
        target_names=("first", "second"),
        validation_block_ids=("V00", "V01"),
        validation_block_sizes=(30, 54),
        block_size=42,
        minimum_training_blocks=4,
    )
    assert len(rows) == 2 * 2 * 4
    assert {row["target"] for row in rows} == {"first", "second"}
    assert {row["block"] for row in rows} == {"V00", "V01"}
    assert {int(row["sample_count"]) for row in rows} == {30, 54}
    ridge = [row for row in rows if row["probe"] == "context_ridge"]
    assert min(float(row["R2"]) for row in ridge) > 0.99


def test_causal_neighbors_exclude_nearby_and_future_samples() -> None:
    train_time = np.arange(100)
    query_time = np.asarray([120, 130])
    train = train_time[:, None].astype(float)
    query = query_time[:, None].astype(float)
    indices, _ = nearest_preceding_neighbors(
        train, train_time, query, query_time, k=5, minimum_separation=42
    )
    assert np.all(train_time[indices[0]] <= 78)
    assert np.all(train_time[indices[1]] <= 88)


def test_b5_shuffle_targets_and_mismatch_are_fixed_and_bijective() -> None:
    assert len(SELECTED_TARGETS) == 18
    assert len(set(SELECTED_TARGETS)) == 18
    mismatches = tuple(mismatched_target(target) for target in SELECTED_TARGETS)
    assert len(set(mismatches)) == 18
    assert all(498 <= value < 624 for value in mismatches)
    assert all((mismatched_target(target) - target) % 126 == 63 for target in SELECTED_TARGETS)


def test_memberwise_nonlinear_reduction_is_not_mean_field_substitution() -> None:
    members = np.asarray([-1.0, 1.0])
    memberwise_then_mean = np.mean(members**2)
    mean_then_nonlinear = np.mean(members) ** 2
    assert memberwise_then_mean == 1.0
    assert mean_then_nonlinear == 0.0
