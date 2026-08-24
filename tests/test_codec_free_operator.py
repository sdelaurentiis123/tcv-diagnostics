"""Known-answer tests for the codec-free state operator and pair planner."""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from tcv_diagnostics.models.codec_free_operator import (
    CodecFreeIncrementOperator3D,
    CodecFreeOperatorConfig,
    StateDerivativePrediction,
    component_balanced_state_derivative_loss,
    normalized_error_metrics,
    persistence_normalized_state_derivative_loss,
    spatialize_boundary,
    state_derivative_loss,
    xy_upsample,
)
from tcv_diagnostics.state_operator_data import (
    LeadTimeStateDataset,
    fit_training_derivative_rms,
    plan_lead_pairs,
)


def tiny_config(*, family: str = "e6b", history: int = 1):
    return CodecFreeOperatorConfig(
        state_family=family,
        history_frames=history,
        base_channels=4,
        channel_multipliers=(1, 2),
        blocks_per_level=1,
        lead_embedding_channels=8,
        group_norm_maximum_groups=2,
        predict_boundary=family == "e6b",
    )


def test_pair_planner_stays_inside_frozen_splits() -> None:
    train = plan_lead_pairs(
        split="train", lead_steps=(1, 2, 4, 8, 16), history_frames=2
    )
    validation = plan_lead_pairs(
        split="validation", lead_steps=(1, 16), history_frames=1
    )
    assert train[0].current == 1
    assert train[0].target == 2
    assert max(pair.target for pair in train) == 431
    assert min(pair.current for pair in validation) == 496
    assert max(pair.target for pair in validation) == 623
    assert all(pair.lead in {1, 2, 4, 8, 16} for pair in train)
    with pytest.raises(ValueError, match="source split"):
        plan_lead_pairs(
            split="train",
            lead_steps=(1,),
            history_frames=1,
            current_interval=(431, 433),
        )


@dataclass(frozen=True)
class _TinyShard:
    path: object


class _TinyNormalization:
    @staticmethod
    def encode_volume(fields, values):
        return np.stack(values, axis=0).astype(np.float32)

    @staticmethod
    def encode_boundary(values):
        return np.asarray(values, dtype=np.float32)


class _TinyCatalog:
    def __init__(self, path):
        self.shard = _TinyShard(path)
        self._verified = {path}
        self.normalization = _TinyNormalization()
        self.verified_frames = None

    def locate(self, frame):
        return self.shard, int(frame)

    def verify_consumed_frames(self, frames):
        self.verified_frames = tuple(int(frame) for frame in frames)
        return (self.shard.path,)


def test_e6b_lead_dataset_returns_joint_volume_and_boundary_derivatives(
    tmp_path,
) -> None:
    path = tmp_path / "development_85604_state.h5"
    fields = ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort")
    with h5py.File(path, "x") as handle:
        coordinates = handle.create_group("coordinates")
        coordinates.create_dataset("frame_index", data=np.arange(6))
        volume = handle.create_group("fields")
        z_pattern = np.arange(6, dtype=np.float32).reshape(1, 1, 6)
        for channel, field in enumerate(fields):
            values = np.stack(
                [
                    np.broadcast_to(z_pattern + frame + channel, (4, 3, 6))
                    for frame in range(6)
                ],
                axis=0,
            )
            volume.create_dataset(field, data=values)
        boundary = handle.create_group("boundary")
        boundary.create_dataset(
            "Bphi",
            data=np.stack(
                [np.full((2, 3), frame, dtype=np.float32) for frame in range(6)]
            ),
        )

    catalog = _TinyCatalog(path)
    dataset = LeadTimeStateDataset(
        catalog,
        family="e6b",
        split="train",
        lead_steps=(1,),
        history_frames=2,
        augment=False,
        seed=1701,
        current_interval=(2, 3),
    )
    item = dataset[0]
    np.testing.assert_array_equal(item["context_frame_indices"], [1, 2])
    assert int(item["target_frame_index"]) == 3
    assert float(item["lead_steps"]) == 1.0
    np.testing.assert_allclose(item["target_derivative"], 1.0)
    np.testing.assert_allclose(item["target_boundary_derivative"], 1.0)
    assert item["context"].shape == (2, 6, 4, 3, 6)
    assert item["context_boundary"].shape == (2, 2, 3)
    assert catalog.verified_frames == (1, 2, 3)
    dataset.close()


def test_boundary_spatialization_places_values_only_at_radial_sides() -> None:
    boundary = torch.tensor([[[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]]])
    fields = spatialize_boundary(boundary, n_x=4, n_z=5)
    assert fields.shape == (1, 4, 4, 3, 5)
    torch.testing.assert_close(fields[0, 0, 0, :, 0], boundary[0, 0, 0])
    torch.testing.assert_close(fields[0, 1, -1, :, 0], boundary[0, 0, 1])
    assert torch.count_nonzero(fields[0, 0, 1:]) == 0
    assert torch.count_nonzero(fields[0, 1, :-1]) == 0
    assert torch.all(fields[0, 2, 0] == 1)
    assert torch.all(fields[0, 3, -1] == 1)


@pytest.mark.parametrize("shift", [1, 3, 7])
def test_e6b_operator_is_equivariant_to_integer_toroidal_rolls(shift: int) -> None:
    torch.manual_seed(11)
    model = CodecFreeIncrementOperator3D(tiny_config()).eval()
    context = torch.randn(2, 1, 6, 8, 6, 12)
    boundary = torch.randn(2, 1, 2, 6)
    leads = torch.tensor([1.0, 4.0])
    with torch.inference_mode():
        reference = model(context, leads, boundary)
        shifted = model(torch.roll(context, shift, -1), leads, boundary)
    torch.testing.assert_close(
        shifted.volume,
        torch.roll(reference.volume, shift, -1),
        atol=3e-6,
        rtol=3e-6,
    )
    torch.testing.assert_close(
        shifted.boundary, reference.boundary, atol=3e-6, rtol=3e-6
    )


def test_c5p_and_e6b_share_processor_contracts() -> None:
    c5p = CodecFreeIncrementOperator3D(tiny_config(family="c5p"))
    e6b = CodecFreeIncrementOperator3D(tiny_config(family="e6b"))
    assert c5p.config.level_channels == e6b.config.level_channels
    assert c5p.to_record()["architecture"]["latent_codec"] is False
    assert e6b.to_record()["architecture"]["physics_derived_loss"] is False
    assert c5p.boundary_head is None
    assert e6b.boundary_head is not None


def test_operator_never_strides_or_resizes_toroidal_axis() -> None:
    model = CodecFreeIncrementOperator3D(tiny_config())
    strides = [
        tuple(module.stride)
        for module in model.modules()
        if isinstance(module, nn.Conv3d)
    ]
    assert strides
    assert all(stride[-1] == 1 for stride in strides)
    values = torch.randn(1, 3, 4, 3, 11)
    torch.testing.assert_close(
        xy_upsample(torch.roll(values, 4, -1), (8, 6, 11)),
        torch.roll(xy_upsample(values, (8, 6, 11)), 4, -1),
    )
    with pytest.raises(ValueError, match="cannot resize"):
        xy_upsample(values, (8, 6, 22))


def test_forecast_uses_lead_scaled_derivative_and_predicts_boundary() -> None:
    torch.manual_seed(13)
    model = CodecFreeIncrementOperator3D(tiny_config()).eval()
    context = torch.randn(2, 1, 6, 8, 6, 12)
    boundary = torch.randn(2, 1, 2, 6)
    leads = torch.tensor([1.0, 8.0])
    with torch.inference_mode():
        derivative = model(context, leads, boundary)
        forecast = model.forecast(context, leads, boundary)
    torch.testing.assert_close(
        forecast.volume,
        context[:, -1] + leads[:, None, None, None, None] * derivative.volume,
    )
    assert derivative.boundary is not None and forecast.boundary is not None
    torch.testing.assert_close(
        forecast.boundary,
        boundary[:, -1] + leads[:, None, None] * derivative.boundary,
    )


def test_state_loss_uses_only_direct_state_targets() -> None:
    volume = torch.zeros(2, 6, 4, 3, 5, requires_grad=True)
    boundary = torch.zeros(2, 2, 3, requires_grad=True)
    prediction = StateDerivativePrediction(volume=volume, boundary=boundary)
    loss = state_derivative_loss(
        prediction,
        torch.ones_like(volume),
        torch.full_like(boundary, 2.0),
    )
    torch.testing.assert_close(loss, torch.tensor(5.0))
    loss.backward()
    assert volume.grad is not None
    assert boundary.grad is not None


def test_component_balanced_loss_weights_each_field_and_side_equally() -> None:
    volume = torch.zeros(1, 2, 2, 1, 1, requires_grad=True)
    boundary = torch.zeros(1, 2, 3, requires_grad=True)
    target_volume = torch.stack(
        (torch.ones(1, 2, 1, 1), torch.full((1, 2, 1, 1), 2.0)), dim=1
    )
    target_boundary = torch.stack(
        (torch.full((1, 3), 3.0), torch.full((1, 3), 4.0)), dim=1
    )
    loss, records = component_balanced_state_derivative_loss(
        StateDerivativePrediction(volume=volume, boundary=boundary),
        target_volume,
        target_boundary,
    )
    torch.testing.assert_close(loss, torch.tensor((1.0 + 4.0 + 9.0 + 16.0) / 4.0))
    torch.testing.assert_close(records["volume_mean"], torch.tensor(2.5))
    torch.testing.assert_close(records["boundary_mean"], torch.tensor(12.5))
    loss.backward()
    assert volume.grad is not None
    assert boundary.grad is not None


def test_component_balanced_c5p_loss_rejects_boundary_target() -> None:
    prediction = StateDerivativePrediction(
        volume=torch.zeros(1, 5, 2, 2, 2), boundary=None
    )
    loss, records = component_balanced_state_derivative_loss(
        prediction, torch.ones_like(prediction.volume)
    )
    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(records["total"], loss)
    with pytest.raises(ValueError, match="boundary target"):
        component_balanced_state_derivative_loss(
            prediction,
            torch.ones_like(prediction.volume),
            torch.ones(1, 2, 2),
        )


def test_persistence_normalized_loss_equalizes_component_target_rms() -> None:
    volume = torch.zeros(1, 2, 2, 1, 1, requires_grad=True)
    boundary = torch.zeros(1, 2, 3, requires_grad=True)
    target_volume = torch.stack(
        (torch.ones(1, 2, 1, 1), torch.full((1, 2, 1, 1), 2.0)), dim=1
    )
    target_boundary = torch.stack(
        (torch.full((1, 3), 3.0), torch.full((1, 3), 4.0)), dim=1
    )
    loss, records = persistence_normalized_state_derivative_loss(
        StateDerivativePrediction(volume=volume, boundary=boundary),
        target_volume,
        torch.tensor([1.0, 2.0]),
        target_boundary,
        torch.tensor([3.0, 4.0]),
    )
    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(records["volume_mean"], torch.tensor(1.0))
    torch.testing.assert_close(records["boundary_mean"], torch.tensor(1.0))
    loss.backward()
    assert volume.grad is not None and boundary.grad is not None


def test_zero_initialized_operator_starts_at_persistence() -> None:
    config = CodecFreeOperatorConfig(
        state_family="e6b",
        history_frames=1,
        base_channels=4,
        channel_multipliers=(1, 2),
        blocks_per_level=1,
        lead_embedding_channels=8,
        group_norm_maximum_groups=2,
        predict_boundary=True,
        zero_initialize_output=True,
    )
    model = CodecFreeIncrementOperator3D(config).eval()
    context = torch.randn(2, 1, 6, 8, 6, 12)
    boundary = torch.randn(2, 1, 2, 6)
    with torch.inference_mode():
        derivative = model(context, torch.tensor([1.0, 4.0]), boundary)
        forecast = model.forecast(context, torch.tensor([1.0, 4.0]), boundary)
    torch.testing.assert_close(derivative.volume, torch.zeros_like(derivative.volume))
    assert derivative.boundary is not None
    torch.testing.assert_close(
        derivative.boundary, torch.zeros_like(derivative.boundary)
    )
    torch.testing.assert_close(forecast.volume, context[:, -1])
    assert forecast.boundary is not None
    torch.testing.assert_close(forecast.boundary, boundary[:, -1])


def test_training_derivative_rms_is_fit_from_direct_state_targets(tmp_path) -> None:
    path = tmp_path / "development_85604_state_rms.h5"
    fields = ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort")
    increments = np.arange(1, 7, dtype=np.float32)
    with h5py.File(path, "x") as handle:
        coordinates = handle.create_group("coordinates")
        coordinates.create_dataset("frame_index", data=np.arange(6))
        volume = handle.create_group("fields")
        for channel, field in enumerate(fields):
            volume.create_dataset(
                field,
                data=np.stack(
                    [
                        np.full((2, 2, 3), frame * increments[channel], dtype=np.float32)
                        for frame in range(6)
                    ]
                ),
            )
        boundary = handle.create_group("boundary")
        boundary.create_dataset(
            "Bphi",
            data=np.stack(
                [
                    np.stack(
                        (
                            np.full(2, 7.0 * frame, dtype=np.float32),
                            np.full(2, 8.0 * frame, dtype=np.float32),
                        )
                    )
                    for frame in range(6)
                ]
            ),
        )
    dataset = LeadTimeStateDataset(
        _TinyCatalog(path),
        family="e6b",
        split="train",
        lead_steps=(1,),
        history_frames=1,
        augment=False,
        seed=1701,
        current_interval=(1, 4),
    )
    fitted = fit_training_derivative_rms(dataset)
    np.testing.assert_allclose(fitted.volume, increments)
    np.testing.assert_allclose(fitted.boundary, [7.0, 8.0])
    assert fitted.pair_count == 3
    assert fitted.to_record()["fit_split"] == "train"
    dataset.close()


def test_normalized_error_metrics_have_known_maximum_and_rms() -> None:
    reference = torch.zeros(4, dtype=torch.float32)
    candidate = torch.tensor([0.0, 0.0, 0.0, 2.0], dtype=torch.float32)
    metrics = normalized_error_metrics(candidate, reference)
    assert metrics["normalized_maximum_absolute_error"] == 2.0
    assert metrics["normalized_root_mean_square_error"] == 1.0
    assert normalized_error_metrics(reference, reference) == {
        "normalized_maximum_absolute_error": 0.0,
        "normalized_root_mean_square_error": 0.0,
    }


def test_family_boundary_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="only E6B"):
        CodecFreeOperatorConfig(state_family="c5p", predict_boundary=True)
    model = CodecFreeIncrementOperator3D(tiny_config(family="c5p"))
    context = torch.randn(1, 1, 5, 8, 6, 12)
    with pytest.raises(ValueError, match="does not accept"):
        model(context, torch.ones(1), torch.randn(1, 1, 2, 6))
