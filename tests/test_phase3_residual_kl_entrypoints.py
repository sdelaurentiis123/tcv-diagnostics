"""Fail-closed source and small-array checks for residual-KL entry points."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from paper0.tools.evaluate_residual_kl_oracle import (
    _available_ranks,
    _validation_variance_capture,
    _write_projection,
)


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "paper0/tools/build_residual_kl_pretruth.py"
EVALUATOR = ROOT / "paper0/tools/evaluate_residual_kl_oracle.py"


def test_pretruth_builder_never_opens_validation_truth_or_model_code() -> None:
    source = BUILDER.read_text(encoding="utf-8")
    assert 'split="train"' in source
    assert 'split="validation"' not in source
    assert "NativeTruthCatalog" not in source
    assert "torch" not in source
    assert "--checkpoint" not in source
    assert '"validation_truth_read": False' in source
    assert '"model_inference_performed": False' in source
    assert '"covariance_empirical_mean_added_to_forecast": False' in source
    assert "training_covariance_reference" in source


def test_evaluator_rehashes_closed_pretruth_before_first_validation_truth() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    closure = source.index("pretruth = _validate_pretruth_closure(")
    marker = source.index("# This is the first validation-truth read")
    truth = source.index("validation_truth = _load_validation_truth(catalog)")
    assert closure < marker < truth
    assert "verified_closed_pretruth_before_validation" in source
    assert "training_covariance_reference_used_centered_KL_matrix_R" in source
    assert 'validation_blocks=tuple(' in source
    assert '"chronological_blocks": chronological' in source
    assert "checkpoint_loaded\": False" in source
    assert "model_training_performed\": False" in source


def test_available_rank_ladder_never_clips_unavailable_ranks() -> None:
    assert _available_ranks(20) == [0, 8, 16, 20]
    assert _available_ranks(429) == [0, 8, 16, 32, 44, 64, 128, 256, 429]


def test_streamed_projection_is_nested_and_rank_zero_is_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(81)
    modes, _ = np.linalg.qr(rng.normal(size=(20, 4)))
    modes = modes.T
    values = rng.normal(size=(3, 20))
    coefficients = values @ modes.T
    captures = []
    destination = np.memmap(
        tmp_path / "projection.dat", mode="w+", dtype=np.float32, shape=(3, 5, 1, 1, 4)
    )
    reference = values.reshape(3, 5, 1, 1, 4).astype(np.float32)
    for rank in (0, 1, 2, 4):
        _write_projection(
            coefficients=coefficients,
            modes=modes,
            rank=rank,
            destination=destination,
        )
        if rank == 0:
            assert np.count_nonzero(destination) == 0
        captures.append(
            _validation_variance_capture(reference, np.asarray(destination))["total"]
        )
    assert np.all(np.diff(captures) >= -1e-7)
