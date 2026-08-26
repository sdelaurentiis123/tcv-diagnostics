"""Known-answer checks for generic C5P/E6B bounded state composition."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tcv_diagnostics.state_view_rollout import (
    autoregressive_state_forecast_path,
    direct_state_forecast,
)


class IncrementModel:
    def __init__(self, family: str) -> None:
        self.family = family
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def forecast(
        self,
        context: torch.Tensor,
        lead: torch.Tensor,
        context_boundary: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        self.calls.append(
            (
                context[:, -1].clone(),
                None
                if context_boundary is None
                else context_boundary[:, -1].clone(),
            )
        )
        volume = context[:, -1] + lead.reshape(-1, 1, 1, 1, 1)
        boundary = None
        if self.family == "e6b":
            assert context_boundary is not None
            boundary = context_boundary[:, -1] + lead.reshape(-1, 1, 1)
        return SimpleNamespace(volume=volume, boundary=boundary)


@pytest.mark.parametrize("family,channels", [("c5p", 5), ("e6b", 6)])
def test_direct_and_autoregressive_paths_match_known_increments(
    family: str, channels: int
) -> None:
    model = IncrementModel(family)
    volume = torch.zeros((2, channels, 4, 3, 5))
    boundary = torch.zeros((2, 2, 3)) if family == "e6b" else None

    direct_volume, direct_boundary = direct_state_forecast(
        model,
        volume,
        boundary,
        family=family,
        horizon=4,
    )
    assert torch.all(direct_volume == 4)
    if family == "e6b":
        assert direct_boundary is not None
        assert torch.all(direct_boundary == 4)
    else:
        assert direct_boundary is None

    model.calls.clear()
    path = autoregressive_state_forecast_path(
        model,
        volume,
        boundary,
        family=family,
        step=1,
        horizon=4,
    )
    assert len(path) == 4
    assert torch.all(path[-1][0] == 4)
    assert torch.all(model.calls[1][0] == 1)
    assert torch.all(model.calls[2][0] == 2)
    assert torch.all(model.calls[3][0] == 3)
    if family == "e6b":
        assert path[-1][1] is not None
        assert torch.all(path[-1][1] == 4)


def test_state_family_and_boundary_contracts_fail_closed() -> None:
    c5p = IncrementModel("c5p")
    with pytest.raises(ValueError, match="C5P"):
        direct_state_forecast(
            c5p,
            torch.zeros((1, 5, 2, 3, 4)),
            torch.zeros((1, 2, 3)),
            family="c5p",
            horizon=1,
        )
    e6b = IncrementModel("e6b")
    with pytest.raises(ValueError, match="E6B"):
        direct_state_forecast(
            e6b,
            torch.zeros((1, 6, 2, 3, 4)),
            None,
            family="e6b",
            horizon=1,
        )
