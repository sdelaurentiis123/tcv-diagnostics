import torch
from torch import nn

from tcv_diagnostics.models.o2 import (
    C5POneStepModel,
    MaskedLatentTransition,
    O2ViTConfig,
)
from tcv_diagnostics.models.vit import (
    MultiheadSelfAttention,
    ViT,
    apply_rope,
    regular_grid_coordinates,
)


class ToyCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def encode(self, values):
        return values * self.scale

    def decode(self, values):
        return values * self.scale


def tiny_config(**overrides):
    values = {
        "latent_channels": 2,
        "hidden_channels": 8,
        "transformer_blocks": 2,
        "attention_heads": 2,
        "ffn_factor": 2,
        "latent_patch": (1, 1, 1),
        "qk_normalization": True,
        "rope": True,
        "dropout": 0.0,
        "activation_checkpointing": False,
    }
    values.update(overrides)
    return O2ViTConfig(**values)


def test_regular_coordinates_follow_flattened_grid_order():
    coordinates = regular_grid_coordinates(
        (2, 2), dtype=torch.float32, device=torch.device("cpu")
    )
    assert torch.equal(
        coordinates,
        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32),
    )


def test_rope_zero_angle_is_identity_and_rotation_preserves_norm():
    generator = torch.Generator().manual_seed(4)
    query = torch.randn(2, 2, 5, 4, generator=generator)
    key = torch.randn(2, 2, 5, 4, generator=generator)
    zero = torch.zeros(1, 2, 5, 2)
    query_zero, key_zero = apply_rope(query, key, zero)
    assert torch.equal(query_zero, query)
    assert torch.equal(key_zero, key)

    theta = torch.randn(1, 2, 5, 2, generator=generator)
    query_rotated, key_rotated = apply_rope(query, key, theta)
    assert torch.allclose(query_rotated.norm(dim=-1), query.norm(dim=-1))
    assert torch.allclose(key_rotated.norm(dim=-1), key.norm(dim=-1))


def test_qk_attention_shape_and_finite_gradient():
    attention = MultiheadSelfAttention(8, attention_heads=2, qk_norm=True)
    tokens = torch.randn(2, 7, 8, requires_grad=True)
    theta = torch.randn(7, 4)
    output = attention(tokens, theta)
    assert output.shape == tokens.shape
    output.square().mean().backward()
    assert tokens.grad is not None
    assert torch.all(torch.isfinite(tokens.grad))


def test_four_dimensional_vit_patch_roundtrip_shape():
    model = ViT(
        2,
        3,
        condition_channels=1,
        hidden_channels=8,
        hidden_blocks=1,
        attention_heads=2,
        ffn_factor=2,
        spatial=4,
        patch_size=(1, 2, 2, 1),
        dropout=0.0,
        checkpointing=False,
    )
    values = torch.randn(2, 2, 3, 4, 2, 5)
    mask = torch.ones(2, 1, 3, 4, 2, 5)
    output = model(values, mask)
    assert output.shape == (2, 3, 3, 4, 2, 5)


def test_masked_trajectory_preserves_history_order_and_zeros_target():
    transition = MaskedLatentTransition(context_frames=2, config=tiny_config())
    context = torch.stack(
        (torch.full((2, 2, 3, 2, 4), 11.0), torch.full((2, 2, 3, 2, 4), 22.0)),
        dim=1,
    )
    trajectory, mask = transition.masked_trajectory(context)
    assert trajectory.shape == (2, 3, 2, 3, 2, 4)
    assert torch.equal(trajectory[:, 0], context[:, 0])
    assert torch.equal(trajectory[:, 1], context[:, 1])
    assert torch.count_nonzero(trajectory[:, 2]) == 0
    assert torch.count_nonzero(mask[:, :2] != 1) == 0
    assert torch.count_nonzero(mask[:, 2]) == 0


def test_h1_and_h2_have_identical_parameterization_and_seeded_initialization():
    config = tiny_config()
    torch.manual_seed(1701)
    h1 = MaskedLatentTransition(context_frames=1, config=config)
    torch.manual_seed(1701)
    h2 = MaskedLatentTransition(context_frames=2, config=config)
    assert h1.state_dict().keys() == h2.state_dict().keys()
    for key in h1.state_dict():
        assert torch.equal(h1.state_dict()[key], h2.state_dict()[key]), key


def test_frozen_configuration_has_704_tokens_per_frame():
    transition = MaskedLatentTransition(context_frames=2)
    latent = torch.zeros(1, 32, 3, 16, 8, 22)
    condition = torch.zeros(1, 1, 3, 16, 8, 22)
    patches = transition.backbone.patchify(torch.cat((latent, condition), dim=1))
    assert patches.shape[2:] == (3, 8, 4, 22)
    assert patches[0].numel() // patches.shape[1] == 3 * 704


def test_zero_increment_gives_latest_frame_and_canonical_deterministic_members():
    transition = MaskedLatentTransition(context_frames=2, config=tiny_config())
    for parameter in transition.parameters():
        nn.init.zeros_(parameter)
    model = C5POneStepModel(
        codec=ToyCodec(),
        transition=transition,
        latent_mean=torch.tensor([0.0, 0.0]),
        latent_standard_deviation=torch.tensor([1.0, 1.0]),
    )
    context = torch.randn(3, 2, 2, 3, 2, 4)
    forecast = model(context)
    assert torch.equal(forecast, context[:, -1])
    ensemble = model.predict(context, horizon=1, ensemble_size=4)
    assert ensemble.shape == (3, 4, 1, 2, 3, 2, 4)
    for member in range(4):
        assert torch.equal(ensemble[:, member, 0], context[:, -1])


def test_codec_stays_frozen_while_field_loss_reaches_transition():
    transition = MaskedLatentTransition(context_frames=1, config=tiny_config())
    codec = ToyCodec()
    model = C5POneStepModel(
        codec=codec,
        transition=transition,
        latent_mean=torch.tensor([0.2, -0.3]),
        latent_standard_deviation=torch.tensor([0.7, 1.4]),
    ).train()
    context = torch.randn(2, 1, 2, 3, 2, 4)
    target = torch.randn(2, 2, 3, 2, 4)
    loss = (model(context) - target).abs().mean()
    loss.backward()
    assert model.codec.training is False
    assert codec.scale.requires_grad is False
    assert codec.scale.grad is None
    gradients = [parameter.grad for parameter in transition.parameters()]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        torch.all(torch.isfinite(gradient))
        for gradient in gradients
        if gradient is not None
    )


def test_o2_predict_rejects_unauthorized_horizons():
    model = C5POneStepModel(
        codec=ToyCodec(),
        transition=MaskedLatentTransition(context_frames=1, config=tiny_config()),
        latent_mean=torch.zeros(2),
        latent_standard_deviation=torch.ones(2),
    )
    context = torch.randn(1, 1, 2, 2, 2, 2)
    try:
        model.predict(context, horizon=2, ensemble_size=1)
    except ValueError as error:
        assert "one-step" in str(error)
    else:
        raise AssertionError("O2 accepted an unauthorized multi-step request")
