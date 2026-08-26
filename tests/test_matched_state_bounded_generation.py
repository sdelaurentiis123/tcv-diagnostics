"""Known-answer checks for causal matched-state bounded generation."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from paper0.tools.generate_matched_state_bounded_forecasts import (
    E6BCandidateWriter,
    NATIVE_SHAPE,
    reset_allocated_cuda_peak_memory,
)
from tcv_diagnostics.matched_codec_evaluation import NATIVE81_FIELDS


def _native(count: int) -> dict[str, np.ndarray]:
    return {
        field: np.full((count, *NATIVE_SHAPE), index + 1, dtype=np.float32)
        for index, field in enumerate(NATIVE81_FIELDS["e6b"])
    }


def test_e6b_candidate_contains_only_predictions_needed_by_elliptic_solve(
    tmp_path,
) -> None:
    path = tmp_path / "candidate.h5"
    writer = E6BCandidateWriter(
        path,
        target_frames=np.asarray([500, 501], dtype=np.int64),
        method="direct",
    )
    writer.write(
        0,
        native_fields=_native(2),
        boundary=np.ones((2, 2, 32), dtype=np.float32),
    )
    writer.finish()

    with h5py.File(path, "r") as handle:
        assert handle.attrs["development_run"] == "85604"
        assert not bool(handle.attrs["held_out_85606_read"])
        assert not bool(handle.attrs["new_nersc_data_read"])
        assert not bool(handle.attrs["target_truth_used_during_generation"])
        assert handle.attrs["boundary_policy"] == (
            "predicted_Bphi_no_truth_bypass"
        )
        assert set(handle["candidate"]) == set(NATIVE81_FIELDS["e6b"])
        assert "phi" not in handle["candidate"]
        assert np.array_equal(handle["coordinates/frame_index"][:], [500, 501])
        assert handle["boundary/Bphi"].shape == (2, 2, 32)


def test_e6b_candidate_refuses_partial_or_overlapping_output(tmp_path) -> None:
    partial_path = tmp_path / "partial.h5"
    partial = E6BCandidateWriter(
        partial_path,
        target_frames=np.asarray([504, 505], dtype=np.int64),
        method="autoregressive_lead1",
    )
    partial.write(
        0,
        native_fields=_native(1),
        boundary=np.ones((1, 2, 32), dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="every target"):
        partial.finish()
    partial.abort()

    overlap_path = tmp_path / "overlap.h5"
    overlap = E6BCandidateWriter(
        overlap_path,
        target_frames=np.asarray([504, 505], dtype=np.int64),
        method="autoregressive_lead2",
    )
    overlap.write(
        0,
        native_fields=_native(1),
        boundary=np.ones((1, 2, 32), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="overlaps"):
        overlap.write(
            0,
            native_fields=_native(1),
            boundary=np.ones((1, 2, 32), dtype=np.float32),
        )
    overlap.abort()


def test_peak_memory_reset_uses_the_current_slurm_cuda_device(monkeypatch) -> None:
    calls: list[bool] = []

    def reset_without_explicit_device() -> None:
        calls.append(True)

    monkeypatch.setattr(
        "paper0.tools.generate_matched_state_bounded_forecasts."
        "torch.cuda.reset_peak_memory_stats",
        reset_without_explicit_device,
    )
    reset_allocated_cuda_peak_memory()
    assert calls == [True]
