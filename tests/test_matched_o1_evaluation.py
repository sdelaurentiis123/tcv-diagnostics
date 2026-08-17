"""Known-answer orchestration tests for the final matched O1 scorer."""

from __future__ import annotations

import numpy as np

from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES
from tcv_diagnostics.matched_o1_evaluation import (
    evaluate_matched_transport_interval,
)


class _Truth:
    def read(self, start, stop, *, fields):
        values = np.arange(start + 1, stop + 1, dtype=np.float64)[:, None, None, None]
        return {field: values.copy() for field in fields}


class _Candidate:
    family = "c5p"
    frames = tuple(range(496, 624))

    def read_native(self, start, stop):
        values = np.arange(start + 1, stop + 1, dtype=np.float64)[:, None, None, None]
        return {
            "Ne": values.copy(),
            "Pe": values.copy(),
            "Pi": values.copy(),
            "phi": values.copy(),
        }


def _fake_transport(state, _geometry):
    values = np.asarray(state["Ne"], dtype=np.float64).reshape(-1)
    return {
        quantity: {
            "strict_face_contributions": values[:, None],
            "separatrix_wedge": values,
        }
        for quantity in TRANSPORT_QUANTITIES
    }


def test_transport_interval_never_mixes_sixteen_frame_blocks(monkeypatch) -> None:
    monkeypatch.setattr(
        "tcv_diagnostics.matched_o1_evaluation.evaluate_transport_state",
        _fake_transport,
    )
    result = evaluate_matched_transport_interval(
        truth=_Truth(),
        candidate=_Candidate(),
        phi=None,
        geometry=object(),
        split="validation",
        frames=tuple(range(496, 624)),
        chunk_frames=7,
    )
    assert result["frame_count"] == 128
    assert result["overall"]["frames"] == 128
    assert [block["frames"] for block in result["blocks"]] == [16] * 8
    comparison = result["overall"]["comparisons"]["truth_vs_reconstruction"]
    assert comparison["quantities"]["particle"]["separatrix"]["metrics"][
        "relative_l2"
    ] == 0.0
