"""CPU implementation gates for the frozen deterministic codec ladder."""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from tcv_diagnostics.models.dcae import (  # noqa: E402
    AutoEncoder,
    CODEC_CONFIGS,
    DCDecoder,
    DCEncoder,
    equal_channel_mae,
    latent_shape,
    normalize_strides,
)
from tcv_diagnostics.models.layers import (  # noqa: E402
    PatchifyND,
    PerAxisConvNd,
    UnpatchifyND,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0/manifests/phase2_matched_o1_o2_85604.json"


class TestPatchOrdering(unittest.TestCase):
    def test_patchify_unpatchify_is_exact_in_three_dimensions(self) -> None:
        source = torch.arange(2 * 3 * 8 * 6 * 10, dtype=torch.float32).reshape(
            2, 3, 8, 6, 10
        )
        patch = (2, 3, 5)
        packed = PatchifyND(patch)(source)

        self.assertEqual(tuple(packed.shape), (2, 90, 4, 2, 2))
        torch.testing.assert_close(UnpatchifyND(patch)(packed), source)

    def test_known_patch_channel_order(self) -> None:
        source = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
        packed = PatchifyND((2, 2, 2))(source)

        self.assertEqual(tuple(packed.shape), (1, 8, 1, 1, 1))
        torch.testing.assert_close(packed.flatten(), torch.arange(8.0))


class TestMixedPadding(unittest.TestCase):
    @staticmethod
    def _unit_convolution() -> PerAxisConvNd:
        layer = PerAxisConvNd(
            1,
            1,
            spatial=3,
            kernel_size=3,
            padding=1,
            padding_mode=("zeros", "zeros", "circular"),
            bias=False,
        )
        with torch.no_grad():
            layer.weight.zero_()
        return layer

    def test_z_padding_wraps(self) -> None:
        layer = self._unit_convolution()
        with torch.no_grad():
            layer.weight[0, 0, 1, 1, 0] = 1.0
        source = torch.arange(4, dtype=torch.float32).reshape(1, 1, 1, 1, 4)

        actual = layer(source).flatten()

        torch.testing.assert_close(actual, torch.tensor([3.0, 0.0, 1.0, 2.0]))

    def test_x_padding_is_a_zero_wall(self) -> None:
        layer = self._unit_convolution()
        with torch.no_grad():
            layer.weight[0, 0, 0, 1, 1] = 1.0
        source = torch.tensor([2.0, 7.0]).reshape(1, 1, 2, 1, 1)

        actual = layer(source).flatten()

        torch.testing.assert_close(actual, torch.tensor([0.0, 2.0]))


class TestCodecStructure(unittest.TestCase):
    def test_frozen_latent_shapes(self) -> None:
        self.assertEqual(latent_shape(CODEC_CONFIGS["dcae_l20"]), (64, 8, 4, 22))
        self.assertEqual(latent_shape(CODEC_CONFIGS["dcae_l10"]), (32, 16, 8, 22))

    def test_per_transition_stride_normalization(self) -> None:
        strides = normalize_strides(
            ((2, 2, 2), (2, 2, 1)), spatial=3, transitions=2
        )
        self.assertEqual(strides, ((2, 2, 2), (2, 2, 1)))

    def test_codec_configs_match_frozen_manifest(self) -> None:
        manifest = json.loads(MANIFEST.read_text())
        frozen = {item["name"]: item for item in manifest["codec_candidates"]}
        self.assertEqual(set(CODEC_CONFIGS), set(frozen))
        for name, config in CODEC_CONFIGS.items():
            self.assertEqual(list(config.hidden_channels), frozen[name]["hidden_channels"])
            self.assertEqual(list(config.hidden_blocks), frozen[name]["hidden_blocks"])
            self.assertEqual(config.latent_channels, frozen[name]["latent_channels"])
            self.assertEqual(
                [list(item) for item in config.strides], frozen[name]["strides"]
            )
            self.assertEqual(list(config.expected_latent_grid), frozen[name]["latent_grid"])
            self.assertEqual(list(config.predictor_patch), frozen[name]["predictor_patch"])

    @staticmethod
    def _tiny_codec() -> AutoEncoder:
        common = {
            "hidden_channels": (4, 8, 16),
            "hidden_blocks": (1, 1, 1),
            "stride": ((2, 2, 2), (2, 2, 1)),
            "kernel_size": 3,
            "spatial": 3,
            "patch_size": 1,
            "periodic": (False, False, True),
            "wall_padding_mode": "zeros",
            "pixel_shuffle": True,
            "ffn_factor": 1,
            "dropout": None,
            "checkpointing": False,
            "identity_init": True,
        }
        return AutoEncoder(
            DCEncoder(2, 3, **common),
            DCDecoder(3, 2, **common),
        )

    def test_anisotropic_codec_roundtrip_shape_and_backward(self) -> None:
        torch.manual_seed(1701)
        codec = self._tiny_codec()
        source = torch.randn(2, 2, 8, 4, 6, requires_grad=True)

        reconstruction, latent = codec(source)
        loss = equal_channel_mae(source, reconstruction)
        loss.backward()

        self.assertEqual(tuple(latent.shape), (2, 3, 2, 1, 3))
        self.assertEqual(tuple(reconstruction.shape), tuple(source.shape))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(source.grad)
        self.assertTrue(torch.isfinite(source.grad).all())

    def test_checkpoint_reload_is_identical(self) -> None:
        torch.manual_seed(1702)
        original = self._tiny_codec().eval()
        source = torch.randn(1, 2, 8, 4, 6)
        with torch.no_grad():
            expected = original(source)

        buffer = io.BytesIO()
        torch.save(original.state_dict(), buffer)
        buffer.seek(0)
        restored = self._tiny_codec().eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            actual = restored(source)

        torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)

    def test_softclip2_is_finite_and_bounded(self) -> None:
        latent = torch.tensor([-1.0e30, -5.0, 0.0, 5.0, 1.0e30])
        saturated = AutoEncoder.saturate(latent)

        self.assertTrue(torch.isfinite(saturated).all())
        self.assertLessEqual(float(saturated.abs().max()), 5.0)

    def test_equal_channel_mae_weights_channels_equally(self) -> None:
        target = torch.zeros(1, 2, 1, 1, 2)
        prediction = torch.tensor([0.0, 2.0, 4.0, 4.0]).reshape(1, 2, 1, 1, 2)

        actual = equal_channel_mae(target, prediction)

        self.assertEqual(float(actual), 2.5)


if __name__ == "__main__":
    unittest.main()
