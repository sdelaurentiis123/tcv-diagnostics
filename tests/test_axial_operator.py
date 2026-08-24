"""Known-answer tests for the nonlocal axial state operator."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tcv_diagnostics.models.axial_operator import (
    AxialIncrementOperator3D,
    AxialOperatorConfig,
)


def tiny_config(**updates) -> AxialOperatorConfig:
    values = {
        "state_family": "e6b",
        "history_frames": 1,
        "auxiliary_context_channels": 1,
        "static_context_channels": 2,
        "width": 8,
        "blocks": 1,
        "attention_heads": 2,
        "feedforward_expansion": 2,
        "lead_embedding_channels": 8,
        "group_norm_maximum_groups": 2,
        "kernel_size": 3,
        "predict_boundary": True,
        "zero_initialize_output": False,
    }
    values.update(updates)
    return AxialOperatorConfig(**values)


def inputs():
    context = torch.randn(2, 1, 6, 6, 4, 8)
    boundary = torch.randn(2, 1, 2, 4)
    auxiliary = torch.randn(2, 1, 1, 6, 4, 8)
    static = torch.randn(2, 2, 6, 4, 8)
    leads = torch.tensor([1.0, 4.0])
    return context, boundary, auxiliary, static, leads


def test_axial_operator_predicts_joint_state_and_boundary() -> None:
    torch.manual_seed(21)
    model = AxialIncrementOperator3D(tiny_config()).eval()
    context, boundary, auxiliary, static, leads = inputs()
    with torch.inference_mode():
        derivative = model(context, leads, boundary, auxiliary, static)
        forecast = model.forecast(context, leads, boundary, auxiliary, static)
    assert derivative.volume.shape == context[:, -1].shape
    assert derivative.boundary is not None
    assert derivative.boundary.shape == boundary[:, -1].shape
    torch.testing.assert_close(
        forecast.volume,
        context[:, -1] + leads[:, None, None, None, None] * derivative.volume,
    )
    assert forecast.boundary is not None
    torch.testing.assert_close(
        forecast.boundary,
        boundary[:, -1] + leads[:, None, None] * derivative.boundary,
    )


@pytest.mark.parametrize("shift", [1, 3, 7])
def test_axial_operator_is_toroidally_equivariant(shift: int) -> None:
    torch.manual_seed(22)
    model = AxialIncrementOperator3D(tiny_config()).eval()
    context, boundary, auxiliary, static, leads = inputs()
    with torch.inference_mode():
        reference = model(context, leads, boundary, auxiliary, static)
        shifted = model(
            torch.roll(context, shift, -1),
            leads,
            boundary,
            torch.roll(auxiliary, shift, -1),
            torch.roll(static, shift, -1),
        )
    torch.testing.assert_close(
        shifted.volume,
        torch.roll(reference.volume, shift, -1),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        shifted.boundary,
        reference.boundary,
        atol=1e-5,
        rtol=1e-5,
    )


def test_axial_operator_has_no_toroidal_stride_or_absolute_coordinate() -> None:
    model = AxialIncrementOperator3D(tiny_config())
    convolutions = [
        module for module in model.modules() if isinstance(module, nn.Conv3d)
    ]
    assert convolutions
    assert all(tuple(module.stride)[-1] == 1 for module in convolutions)
    architecture = model.to_record()["architecture"]
    assert architecture["toroidal_downsampling"] is False
    assert architecture["absolute_z_coordinate"] is False
    assert architecture["official_GAOT_reproduction"] is False


def test_axial_operator_has_full_domain_receptive_field() -> None:
    torch.manual_seed(23)
    model = AxialIncrementOperator3D(tiny_config()).eval()
    context, boundary, auxiliary, static, leads = inputs()
    context = context[:1].clone().requires_grad_(True)
    prediction = model(
        context,
        leads[:1],
        boundary[:1],
        auxiliary[:1],
        static[:1],
    )
    prediction.volume[0, 0, 0, 0, 0].backward()
    assert context.grad is not None
    distant_gradient = context.grad[0, 0, :, -1, -1, -1]
    assert torch.count_nonzero(distant_gradient) > 0


def test_axial_operator_zero_initialization_starts_at_persistence() -> None:
    model = AxialIncrementOperator3D(
        tiny_config(zero_initialize_output=True)
    ).eval()
    context, boundary, auxiliary, static, leads = inputs()
    with torch.inference_mode():
        forecast = model.forecast(context, leads, boundary, auxiliary, static)
    torch.testing.assert_close(forecast.volume, context[:, -1])
    assert forecast.boundary is not None
    torch.testing.assert_close(forecast.boundary, boundary[:, -1])


def test_axial_operator_context_contracts_fail_closed() -> None:
    model = AxialIncrementOperator3D(tiny_config())
    context, boundary, auxiliary, static, leads = inputs()
    with pytest.raises(ValueError, match="auxiliary context shape"):
        model(context, leads, boundary, None, static)
    with pytest.raises(ValueError, match="static context shape"):
        model(context, leads, boundary, auxiliary, None)


def test_axial_operator_configuration_rejects_invalid_head_partition() -> None:
    with pytest.raises(ValueError, match="divisible"):
        tiny_config(width=10, attention_heads=4)
