"""Known-answer tests for the separate LOLA noise-modulated ViT."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

from tcv_diagnostics.models.modulated_vit import (
    ModulatedViT,
    ModulatedViTBlock,
    NoiseTimeEmbedding,
)


ROOT = Path(__file__).resolve().parents[1]


def test_modulated_vit_does_not_modify_the_completed_o2_vit() -> None:
    path = ROOT / "src/tcv_diagnostics/models/vit.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "0d1a6863f399fe43c57b7bc4b8b52f29d6baf48d59b33d8c9529b51f6843d853"
    )


def test_noise_time_embedding_has_one_global_vector_per_sample() -> None:
    embedding = NoiseTimeEmbedding(8)
    times = torch.tensor([-3.0, 0.0, 2.0])
    result = embedding(times)
    assert result.shape == (3, 8)
    assert torch.isfinite(result).all()
    assert not torch.equal(result[0], result[1])
    with pytest.raises(ValueError, match="shape"):
        embedding(times[:, None])


def test_modulated_block_requires_matched_batch_conditioning() -> None:
    block = ModulatedViTBlock(
        16,
        modulation_features=8,
        ffn_factor=2,
        spatial=4,
        attention_heads=2,
        dropout=0.0,
    )
    tokens = torch.randn(2, 12, 16)
    coordinates = torch.randn(12, 4)
    with pytest.raises(ValueError, match="modulation"):
        block(tokens, torch.randn(1, 8), coordinates, tokens)


def test_modulated_vit_shape_gradient_and_reload_identity() -> None:
    torch.manual_seed(31)
    model = ModulatedViT(
        4,
        4,
        condition_channels=1,
        modulation_features=8,
        hidden_channels=16,
        hidden_blocks=2,
        attention_heads=2,
        ffn_factor=2,
        spatial=4,
        patch_size=(1, 2, 1, 1),
        qk_norm=True,
        rope=True,
        dropout=0.0,
        checkpointing=True,
    )
    inputs = torch.randn(2, 4, 3, 4, 2, 2, requires_grad=True)
    modulation = torch.randn(2, 8, requires_grad=True)
    condition = torch.zeros(2, 1, 3, 4, 2, 2)
    condition[:, :, :2] = 1.0
    output = model(inputs, modulation, condition)
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert modulation.grad is not None and torch.isfinite(modulation.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())

    model.eval()
    with torch.no_grad():
        expected = model(inputs.detach(), modulation.detach(), condition)
    reloaded = copy.deepcopy(model)
    reloaded.load_state_dict(model.state_dict(), strict=True)
    reloaded.eval()
    with torch.no_grad():
        actual = reloaded(inputs.detach(), modulation.detach(), condition)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_modulated_vit_rejects_wrong_condition_and_patch_shapes() -> None:
    with pytest.raises(ValueError, match="patch"):
        ModulatedViT(
            2,
            2,
            condition_channels=1,
            modulation_features=8,
            hidden_channels=16,
            hidden_blocks=1,
            spatial=4,
            patch_size=(1, 2, 1),
        )
    model = ModulatedViT(
        2,
        2,
        condition_channels=1,
        modulation_features=8,
        hidden_channels=16,
        hidden_blocks=1,
        attention_heads=2,
        spatial=4,
        patch_size=1,
    )
    inputs = torch.randn(1, 2, 3, 2, 2, 2)
    modulation = torch.randn(1, 8)
    with pytest.raises(ValueError, match="condition"):
        model(inputs, modulation, torch.zeros(1, 2, 3, 2, 2, 2))
