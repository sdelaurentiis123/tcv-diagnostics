"""Known-answer and identity tests for the prospective B3 FGN retrofit."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from tcv_diagnostics.models.functional_noise import (
    C5PFunctionalNoiseOneStepModel,
    FunctionalNoiseConfig,
    FunctionalNoiseMaskedLatentTransition,
    fair_crps,
)
from tcv_diagnostics.models.o2 import (
    C5POneStepModel,
    MaskedLatentTransition,
    O2ViTConfig,
)


class IdentityCodec(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale.reshape(1, -1, 1, 1, 1)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale.reshape(1, -1, 1, 1, 1)


def _tiny_o2_config() -> O2ViTConfig:
    return O2ViTConfig(
        latent_channels=2,
        hidden_channels=8,
        transformer_blocks=2,
        attention_heads=2,
        ffn_factor=2,
        latent_patch=(1, 1, 1),
        qk_normalization=True,
        rope=True,
        dropout=0.0,
        activation_checkpointing=False,
    )


def _tiny_noise_config() -> FunctionalNoiseConfig:
    return FunctionalNoiseConfig(
        raw_noise_features=3,
        embedded_noise_features=6,
        adapter_last_weight_multiplier=1.0e-2,
    )


def _parent_and_retrofit():
    torch.manual_seed(1701)
    parent = MaskedLatentTransition(context_frames=1, config=_tiny_o2_config())
    torch.manual_seed(99)
    retrofit = FunctionalNoiseMaskedLatentTransition(
        context_frames=1,
        config=_tiny_o2_config(),
        noise_config=_tiny_noise_config(),
    )
    audit = retrofit.load_deterministic_state(parent.state_dict())
    return parent, retrofit, audit


def _retrofit_model() -> C5PFunctionalNoiseOneStepModel:
    _, transition, _ = _parent_and_retrofit()
    return C5PFunctionalNoiseOneStepModel(
        codec=IdentityCodec(2),
        transition=transition,
        latent_mean=torch.tensor([0.1, -0.2]),
        latent_standard_deviation=torch.tensor([0.7, 1.3]),
    )


def test_two_member_fair_crps_known_answers() -> None:
    predictions = torch.tensor([0.0, 2.0]).reshape(1, 2, 1, 1)
    target = torch.tensor([1.0]).reshape(1, 1, 1)
    result = fair_crps(predictions, target)
    torch.testing.assert_close(result.accuracy_per_channel, torch.tensor([1.0]))
    torch.testing.assert_close(result.spread_per_channel, torch.tensor([1.0]))
    torch.testing.assert_close(result.per_channel, torch.tensor([0.0]))
    torch.testing.assert_close(result.total, torch.tensor(0.0))

    collapsed = fair_crps(torch.zeros(1, 2, 1, 1), target)
    torch.testing.assert_close(collapsed.total, torch.tensor(1.0))


def test_fair_crps_is_member_permutation_invariant_and_differentiable() -> None:
    predictions = torch.tensor(
        [[[[0.0]], [[2.0]], [[5.0]]]], requires_grad=True
    )
    target = torch.tensor([[[1.0]]])
    first = fair_crps(predictions, target)
    second = fair_crps(predictions[:, [2, 0, 1]], target)
    torch.testing.assert_close(first.total, second.total, rtol=0.0, atol=0.0)
    first.total.backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_fair_crps_rejects_one_member_or_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="at least two"):
        fair_crps(torch.zeros(1, 1, 2, 3), torch.zeros(1, 2, 3))
    with pytest.raises(ValueError, match="target shape"):
        fair_crps(torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, 4))


def test_parent_load_audit_allows_only_new_noise_keys() -> None:
    parent, retrofit, audit = _parent_and_retrofit()
    assert audit.passed
    assert audit.parent_key_count == len(parent.state_dict())
    assert audit.parent_key_count == audit.deterministic_child_key_count
    assert audit.new_noise_keys
    assert all(
        key.startswith("noise_embedding.") or ".noise_adapter." in key
        for key in audit.new_noise_keys
    )
    assert not audit.unexpected_parent_keys
    assert not audit.missing_deterministic_keys

    malformed = dict(parent.state_dict())
    malformed["unexpected.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="load audit failed"):
        retrofit.load_deterministic_state(malformed)


def test_noise_disabled_transition_is_bitwise_identical_to_parent() -> None:
    parent, retrofit, _ = _parent_and_retrofit()
    parent.eval()
    retrofit.eval()
    context = torch.randn(3, 1, 2, 2, 2, 2)
    with torch.inference_mode():
        expected = parent(context)
        observed = retrofit(context, raw_noise=None)
    assert torch.equal(observed, expected)


def test_noise_embedding_is_global_and_shared_by_all_blocks() -> None:
    _, retrofit, _ = _parent_and_retrofit()
    retrofit.eval()
    captured: list[torch.Tensor] = []
    handles = []
    for block in retrofit.backbone.blocks:
        handles.append(
            block.noise_adapter[0].register_forward_pre_hook(
                lambda _module, inputs: captured.append(inputs[0].detach().clone())
            )
        )
    context = torch.randn(2, 1, 2, 2, 2, 2)
    noise = torch.randn(2, 3)
    with torch.inference_mode():
        retrofit(context, noise)
    for handle in handles:
        handle.remove()
    assert len(captured) == 2
    assert all(item.shape == (2, 6) for item in captured)
    assert torch.equal(captured[0], captured[1])


def test_official_adapter_initialization_contract() -> None:
    _, retrofit, _ = _parent_and_retrofit()
    for block in retrofit.backbone.blocks:
        assert torch.count_nonzero(block.noise_adapter[-1].bias) == 0
        assert torch.isfinite(block.noise_adapter[-1].weight).all()
        assert block.noise_adapter[-1].weight.std() < 0.01


def test_canonical_members_are_finite_diverse_and_use_no_future_truth() -> None:
    model = _retrofit_model().eval()
    context = torch.randn(2, 1, 2, 2, 2, 2)
    raw_noise = torch.tensor(
        [
            [[-1.0, 0.0, 1.0], [1.0, 0.5, -0.5], [0.2, -0.3, 0.4]],
            [[0.3, 0.8, -1.2], [-0.7, 1.1, 0.1], [1.4, -0.2, 0.6]],
        ]
    )
    with torch.inference_mode():
        latent = model.standardized_latent_members(context, raw_noise)
        forecast = model.predict_with_noise(context, raw_noise)
    assert latent.shape == (2, 3, 2, 2, 2, 2)
    assert forecast.shape == (2, 3, 1, 2, 2, 2, 2)
    assert torch.isfinite(latent).all()
    assert torch.isfinite(forecast).all()
    assert torch.count_nonzero(latent[:, 0] - latent[:, 1]) > 0
    assert torch.count_nonzero(forecast[:, 0] - forecast[:, 1]) > 0
    assert "target" not in inspect.signature(model.predict).parameters
    with pytest.raises(ValueError, match="one-step"):
        model.predict(context, horizon=2, ensemble_size=2)


def test_decoded_fair_crps_reaches_new_and_common_weights_but_not_codec() -> None:
    model = _retrofit_model().train()
    context = torch.randn(2, 1, 2, 2, 2, 2)
    target = torch.randn(2, 2, 2, 2, 2)
    raw_noise = torch.randn(2, 2, 3)
    predictions = model.predict_with_noise(context, raw_noise)[:, :, 0]
    loss = fair_crps(predictions, target).total
    loss.backward()

    assert model.codec.training is False
    assert all(not parameter.requires_grad for parameter in model.codec.parameters())
    assert all(parameter.grad is None for parameter in model.codec.parameters())
    assert model.transition.backbone.input_projection.weight.grad is not None
    assert torch.isfinite(
        model.transition.backbone.input_projection.weight.grad
    ).all()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.transition.noise_embedding.parameters()
    )
    for block in model.transition.backbone.blocks:
        gradients = [parameter.grad for parameter in block.noise_adapter.parameters()]
        assert any(
            gradient is not None
            and torch.isfinite(gradient).all()
            and torch.count_nonzero(gradient) > 0
            for gradient in gradients
        )


def test_deterministic_model_identity_and_full_retrofit_reload() -> None:
    parent_transition, retrofit_transition, _ = _parent_and_retrofit()
    parent_model = C5POneStepModel(
        codec=IdentityCodec(2),
        transition=parent_transition,
        latent_mean=torch.tensor([0.1, -0.2]),
        latent_standard_deviation=torch.tensor([0.7, 1.3]),
    ).eval()
    retrofit_model = C5PFunctionalNoiseOneStepModel(
        codec=IdentityCodec(2),
        transition=retrofit_transition,
        latent_mean=torch.tensor([0.1, -0.2]),
        latent_standard_deviation=torch.tensor([0.7, 1.3]),
    ).eval()
    context = torch.randn(2, 1, 2, 2, 2, 2)
    with torch.inference_mode():
        expected = parent_model(context)
        observed = retrofit_model(context, raw_noise=None)
    assert torch.equal(observed, expected)

    _, reloaded_transition, _ = _parent_and_retrofit()
    reloaded_transition.load_state_dict(retrofit_transition.state_dict(), strict=True)
    reloaded_model = C5PFunctionalNoiseOneStepModel(
        codec=IdentityCodec(2),
        transition=reloaded_transition,
        latent_mean=torch.tensor([0.1, -0.2]),
        latent_standard_deviation=torch.tensor([0.7, 1.3]),
    ).eval()
    fixed_noise = torch.randn(2, 2, 3)
    with torch.inference_mode():
        first = retrofit_model.predict_with_noise(context, fixed_noise)
        second = reloaded_model.predict_with_noise(context, fixed_noise)
    assert torch.equal(first, second)


def test_frozen_default_noise_configuration_matches_manifest() -> None:
    config = FunctionalNoiseConfig()
    assert config.raw_noise_features == 32
    assert config.embedded_noise_features == 256
    assert config.adapter_last_weight_multiplier == 1.0e-2
