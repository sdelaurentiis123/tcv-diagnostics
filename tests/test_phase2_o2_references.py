import numpy as np
import pytest

from tcv_diagnostics.o2_references import (
    fit_spectral_ar1,
    persistence,
    two_frame_linear_extrapolation,
)


def test_persistence_and_linear_extrapolation_known_answers():
    first = np.full((2, 3, 2, 8), 2.0, dtype=np.float32)
    latest = np.full_like(first, 5.0)
    context = np.stack((first, latest), axis=0)
    np.testing.assert_array_equal(persistence(context), latest)
    np.testing.assert_array_equal(two_frame_linear_extrapolation(context), 8.0)
    with pytest.raises(ValueError, match="exactly two"):
        two_frame_linear_extrapolation(context[-1:])


def test_spectral_ar1_recovers_known_toroidal_translation():
    rng = np.random.default_rng(1701)
    pairs = []
    shift = 3
    for _ in range(5):
        source = rng.normal(size=(3, 4, 2, 16)).astype(np.float32)
        target = np.roll(source, shift, axis=-1)
        pairs.append((source, target))
    model = fit_spectral_ar1(pairs, relative_ridge=0.0)

    probe = rng.normal(size=(3, 4, 2, 16)).astype(np.float32)
    prediction = model.predict(probe)
    np.testing.assert_allclose(
        prediction,
        np.roll(probe, shift, axis=-1),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    expected_phase = np.exp(-2j * np.pi * np.arange(9) * shift / 16)
    np.testing.assert_allclose(
        model.coefficient,
        np.broadcast_to(expected_phase, model.coefficient.shape),
        atol=1.0e-12,
    )


def test_relative_ridge_and_zero_power_modes_are_explicit():
    source = np.asarray([1, 0, -1, 0, 1, 0, -1, 0], dtype=np.float32)[
        None, None, None
    ]
    source = np.broadcast_to(source, (2, 3, 2, 8)).copy()
    target = 2.0 * source
    model = fit_spectral_ar1([(source, target)], relative_ridge=1.0e-2)

    assert np.all(model.denominator[:, 0] == 0.0)
    assert np.all(model.coefficient[:, 0] == 0.0)
    np.testing.assert_allclose(model.coefficient[:, 2], 2.0 / 1.01)
    assert model.to_record()["validation_tuning_used"] is False
    assert model.to_record()["pooled_axes"] == ["time", "x", "y"]


def test_spectral_ar1_rejects_empty_or_inconsistent_training_pairs():
    with pytest.raises(ValueError, match="without training pairs"):
        fit_spectral_ar1([])
    source = np.zeros((2, 3, 2, 8), dtype=np.float32)
    target = np.zeros((2, 3, 2, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="shapes differ"):
        fit_spectral_ar1([(source, target)])
