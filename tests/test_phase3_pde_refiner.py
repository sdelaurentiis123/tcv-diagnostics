"""Known-answer, identity, and interface tests for the B4 PDE-Refiner."""

from __future__ import annotations

import inspect
import math

import pytest
import torch
from torch import nn

from tcv_diagnostics.models.o2 import (
    C5POneStepModel,
    MaskedLatentTransition,
    O2ViTConfig,
)
from tcv_diagnostics.models.pde_refiner import (
    C5PPDERefinerOneStepModel,
    PDERefinerConfig,
    PDERefinerMaskedLatentTransition,
    explicit_denoising_update,
)


class IdentityCodec(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale.reshape(1, -1, 1, 1, 1)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale.reshape(1, -1, 1, 1, 1)


def tiny_o2_config() -> O2ViTConfig:
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


def parent_and_refiner():
    torch.manual_seed(1701)
    parent = MaskedLatentTransition(context_frames=1, config=tiny_o2_config())
    torch.manual_seed(99)
    refiner = PDERefinerMaskedLatentTransition(
        context_frames=1,
        config=tiny_o2_config(),
        refiner_config=PDERefinerConfig(),
    )
    audit = refiner.load_deterministic_state(parent.state_dict())
    return parent, refiner, audit


def refiner_model() -> C5PPDERefinerOneStepModel:
    _, transition, _ = parent_and_refiner()
    return C5PPDERefinerOneStepModel(
        codec=IdentityCodec(2),
        transition=transition,
        latent_mean=torch.tensor([0.1, -0.2]),
        latent_standard_deviation=torch.tensor([0.7, 1.3]),
    )


def test_frozen_schedule_known_answers() -> None:
    config = PDERefinerConfig()
    assert config.levels == (0, 1, 2, 3)
    assert config.minimum_noise_standard_deviation == math.sqrt(4e-7)
    assert config.standard_deviations == (
        0.08583742189325572,
        0.007368062997280775,
        0.0006324555320336759,
    )
    assert config.to_record()["network_calls_per_member"] == 4
    assert config.to_record()["normalized_level_coordinates"] == [
        0.0,
        1 / 3,
        2 / 3,
        1.0,
    ]
    with pytest.raises(ValueError, match="three refinements"):
        PDERefinerConfig(refinement_steps=4)
    with pytest.raises(ValueError, match="4e-7"):
        PDERefinerConfig(minimum_noise_variance=1e-4)


def test_explicit_denoising_algebra_known_answer() -> None:
    noisy = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
    predicted = torch.tensor([[[0.5, -1.0], [2.0, -4.0]]])
    observed = explicit_denoising_update(noisy, predicted, 0.25)
    expected = torch.tensor([[[0.875, -1.75], [2.5, 5.0]]])
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="shapes differ"):
        explicit_denoising_update(noisy, predicted[..., :1], 0.25)
    with pytest.raises(ValueError, match="finite and positive"):
        explicit_denoising_update(noisy, predicted, 0.0)


def test_parent_load_audit_allows_only_new_refinement_keys() -> None:
    parent, refiner, audit = parent_and_refiner()
    assert audit.passed
    assert audit.parent_key_count == len(parent.state_dict())
    assert audit.parent_key_count == audit.deterministic_child_key_count
    assert audit.new_refinement_keys
    assert all(
        key.startswith("level_embedding.") or ".refinement_adapter." in key
        for key in audit.new_refinement_keys
    )
    assert not audit.unexpected_parent_keys
    assert not audit.missing_deterministic_keys

    malformed = dict(parent.state_dict())
    malformed["unexpected.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="load audit failed"):
        refiner.load_deterministic_state(malformed)


def test_actual_level0_zero_candidate_is_bitwise_parent_identical() -> None:
    parent, refiner, _ = parent_and_refiner()
    parent.eval()
    refiner.eval()
    context = torch.randn(3, 1, 2, 2, 2, 2)
    zero = torch.zeros_like(context[:, -1])
    with torch.inference_mode():
        expected = parent(context)
        observed = refiner(context, zero, 0)
    assert torch.equal(observed, expected)


def test_zero_initialized_adapters_and_shared_level_embedding() -> None:
    _, refiner, _ = parent_and_refiner()
    for block in refiner.backbone.blocks:
        assert torch.count_nonzero(block.refinement_adapter[-1].weight) == 0
        assert torch.count_nonzero(block.refinement_adapter[-1].bias) == 0

    captured: list[torch.Tensor] = []
    handles = [
        block.refinement_adapter[0].register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        for block in refiner.backbone.blocks
    ]
    context = torch.randn(2, 1, 2, 2, 2, 2)
    candidate = torch.randn(2, 2, 2, 2, 2)
    with torch.inference_mode():
        refiner(context, candidate, torch.tensor([1, 3]))
    for handle in handles:
        handle.remove()
    assert len(captured) == 2
    assert all(item.shape == (2, 256) for item in captured)
    assert torch.equal(captured[0], captured[1])
    assert torch.count_nonzero(captured[0][0] - captured[0][1]) > 0


def test_provisional_target_slot_and_level_validation_are_strict() -> None:
    _, refiner, _ = parent_and_refiner()
    context = torch.randn(2, 1, 2, 2, 2, 2)
    first = torch.zeros(2, 2, 2, 2, 2)
    second = torch.ones_like(first)
    with torch.inference_mode():
        first_output = refiner(context, first, 1)
        second_output = refiner(context, second, 1)
    assert torch.count_nonzero(first_output - second_output) > 0
    with pytest.raises(ValueError, match="range 0..3"):
        refiner(context, first, 4)
    with pytest.raises(ValueError, match="integer-valued"):
        refiner(context, first, torch.tensor([0.5, 1.0]))
    with pytest.raises(ValueError, match="provisional target shape"):
        refiner(context, first[..., :1], 1)


def test_all_stages_have_canonical_shapes_and_only_final_members_diverge() -> None:
    model = refiner_model().eval()
    context = torch.randn(2, 1, 2, 2, 2, 2)
    generator = torch.Generator(device="cpu").manual_seed(41032)
    noise = torch.randn(2, 3, 3, 2, 2, 2, 2, generator=generator)
    with torch.inference_mode():
        latent = model.standardized_latent_stages(context, noise)
        decoded = model.decoded_stages_with_noise(context, noise)
        forecast = model.predict_with_noise(context, noise)
    assert latent.shape == (2, 3, 4, 2, 2, 2, 2)
    assert decoded.shape == (2, 3, 4, 2, 2, 2, 2)
    assert forecast.shape == (2, 3, 1, 2, 2, 2, 2)
    assert torch.isfinite(latent).all()
    assert torch.isfinite(decoded).all()
    assert torch.equal(latent[:, 0, 0], latent[:, 1, 0])
    assert torch.equal(decoded[:, 0, 0], decoded[:, 2, 0])
    assert torch.count_nonzero(latent[:, 0, -1] - latent[:, 1, -1]) > 0
    assert torch.count_nonzero(forecast[:, 0] - forecast[:, 1]) > 0
    assert torch.equal(forecast[:, :, 0], decoded[:, :, -1])


def test_level0_model_matches_deterministic_parent_model() -> None:
    parent_transition, refiner_transition, _ = parent_and_refiner()
    parent_model = C5POneStepModel(
        codec=IdentityCodec(2),
        transition=parent_transition,
        latent_mean=torch.tensor([0.1, -0.2]),
        latent_standard_deviation=torch.tensor([0.7, 1.3]),
    ).eval()
    model = C5PPDERefinerOneStepModel(
        codec=IdentityCodec(2),
        transition=refiner_transition,
        latent_mean=torch.tensor([0.1, -0.2]),
        latent_standard_deviation=torch.tensor([0.7, 1.3]),
    ).eval()
    context = torch.randn(2, 1, 2, 2, 2, 2)
    noise = torch.randn(2, 2, 3, 2, 2, 2, 2)
    with torch.inference_mode():
        expected = parent_model(context)
        observed = model(context)
        stage0 = model.decoded_stages_with_noise(context, noise)[:, 0, 0]
    assert torch.equal(observed, expected)
    assert torch.equal(stage0, expected)
    assert model.codec.training is False
    assert all(not parameter.requires_grad for parameter in model.codec.parameters())


def test_fixed_noise_and_checkpoint_reload_are_bitwise_exact() -> None:
    model = refiner_model().eval()
    _, restored_transition, _ = parent_and_refiner()
    restored_transition.load_state_dict(model.transition.state_dict(), strict=True)
    restored = C5PPDERefinerOneStepModel(
        codec=IdentityCodec(2),
        transition=restored_transition,
        latent_mean=torch.tensor([0.1, -0.2]),
        latent_standard_deviation=torch.tensor([0.7, 1.3]),
    ).eval()
    context = torch.randn(2, 1, 2, 2, 2, 2)
    noise = torch.randn(2, 2, 3, 2, 2, 2, 2)
    with torch.inference_mode():
        first = model.decoded_stages_with_noise(context, noise)
        second = restored.decoded_stages_with_noise(context, noise)
    assert torch.equal(first, second)


def test_public_predict_is_truth_free_one_step_and_canonical() -> None:
    model = refiner_model().eval()
    context = torch.randn(2, 1, 2, 2, 2, 2)
    with torch.inference_mode():
        forecast = model.predict(context, horizon=1, ensemble_size=3)
    assert forecast.shape == (2, 3, 1, 2, 2, 2, 2)
    assert torch.isfinite(forecast).all()
    assert "target" not in inspect.signature(model.predict).parameters
    with pytest.raises(ValueError, match="one-step"):
        model.predict(context, horizon=2, ensemble_size=2)
    with pytest.raises(ValueError, match="positive"):
        model.predict(context, horizon=1, ensemble_size=0)


def test_level_objectives_reach_parent_and_new_parameters_not_codec() -> None:
    model = refiner_model().train()
    context = torch.randn(2, 1, 2, 2, 2, 2)
    target = torch.randn(2, 2, 2, 2, 2)
    standardized_context = model.encode_context(context)
    standardized_target = model.encode_target(target)

    zero = torch.zeros_like(standardized_target)
    prediction0 = model.transition(standardized_context, zero, 0)
    loss0 = (prediction0 - (standardized_target - standardized_context[:, -1])).square().mean()
    noise = torch.randn_like(standardized_target)
    sigma = model.refinement_standard_deviations[0]
    prediction1 = model.transition(
        standardized_context,
        standardized_target + sigma * noise,
        1,
    )
    loss = loss0 + (prediction1 - noise).square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.codec.parameters())
    parent_gradient = model.transition.backbone.input_projection.weight.grad
    assert parent_gradient is not None and torch.isfinite(parent_gradient).all()
    assert torch.count_nonzero(parent_gradient) > 0
    assert all(
        block.refinement_adapter[-1].weight.grad is not None
        and torch.isfinite(block.refinement_adapter[-1].weight.grad).all()
        and torch.count_nonzero(block.refinement_adapter[-1].weight.grad) > 0
        for block in model.transition.backbone.blocks
    )
