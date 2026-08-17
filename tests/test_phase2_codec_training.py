"""CPU unit gates for the frozen O1 codec optimizer mechanics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tcv_diagnostics.codec_training import (
    CodecRunConfig,
    _validation_loss,
    learning_rate_at_step,
    save_torch_atomic,
)


class _ValidationDataset:
    frames = (496, 497, 498, 499)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        return {
            "volume": np.full((5, 2, 2, 2), float(index), dtype=np.float32),
            "frame_index": np.int64(self.frames[index]),
            "toroidal_roll": np.int64(0),
        }


class _IdentityCodec(nn.Module):
    def forward(self, inputs: torch.Tensor):
        return inputs, inputs[:, :1]


class TestCodecRunConfig(unittest.TestCase):
    def test_full_protocol_is_exact(self) -> None:
        config = CodecRunConfig.frozen(
            mode="full", codec="dcae_l20", family="c5p", seed=1701
        )

        self.assertEqual(config.epochs, 200)
        self.assertEqual(config.train_frames, tuple(range(432)))
        self.assertEqual(config.validation_frames, tuple(range(496, 624)))
        self.assertEqual(config.effective_batch, 16)
        self.assertEqual(config.optimizer_steps_per_epoch, 27)
        self.assertEqual(config.total_optimizer_steps, 5400)
        self.assertEqual(config.warmup_optimizer_steps, 270)
        self.assertEqual(config.channels, 5)
        self.assertFalse(config.to_record()["physics_derived_loss_allowed"])

    def test_smoke_is_bounded_and_exercises_accumulation(self) -> None:
        config = CodecRunConfig.frozen(
            mode="smoke", codec="dcae_l20", family="e6b", seed=1702
        )

        self.assertLessEqual(config.epochs, 2)
        self.assertLessEqual(len(config.train_frames), 16)
        self.assertLessEqual(len(config.validation_frames), 16)
        self.assertEqual(config.gradient_accumulation, 4)
        self.assertEqual(config.channels, 6)
        self.assertEqual(config.total_optimizer_steps, 2)

    def test_unfrozen_choices_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CodecRunConfig.frozen(
                mode="full", codec="dcae_l20", family="c5p", seed=9
            )
        with self.assertRaises(ValueError):
            CodecRunConfig.frozen(
                mode="other", codec="dcae_l20", family="c5p", seed=1701
            )

    def test_learning_rate_has_frozen_boundaries(self) -> None:
        config = CodecRunConfig.frozen(
            mode="full", codec="dcae_l20", family="c5p", seed=1701
        )

        self.assertAlmostEqual(
            learning_rate_at_step(config, 1), config.learning_rate / 270
        )
        self.assertEqual(
            learning_rate_at_step(config, config.warmup_optimizer_steps),
            config.learning_rate,
        )
        self.assertEqual(
            learning_rate_at_step(config, config.total_optimizer_steps),
            config.minimum_learning_rate,
        )
        self.assertLess(
            learning_rate_at_step(config, config.warmup_optimizer_steps + 1),
            config.learning_rate,
        )
        with self.assertRaises(ValueError):
            learning_rate_at_step(config, 0)


class TestCodecTrainingPrimitives(unittest.TestCase):
    def test_validation_identity_is_zero_with_float64_accumulation(self) -> None:
        config = CodecRunConfig.frozen(
            mode="smoke", codec="dcae_l20", family="c5p", seed=1701
        )
        aggregate, per_channel = _validation_loss(
            _IdentityCodec(),
            _ValidationDataset(),
            config,
            torch.device("cpu"),
        )

        self.assertEqual(aggregate, 0.0)
        self.assertEqual(per_channel, [0.0] * 5)

    def test_atomic_checkpoint_replaces_complete_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            save_torch_atomic(path, {"value": torch.tensor([1.0, 2.0])})
            loaded = torch.load(path, weights_only=True)
            torch.testing.assert_close(loaded["value"], torch.tensor([1.0, 2.0]))

            save_torch_atomic(path, {"value": torch.tensor([3.0])})
            replaced = torch.load(path, weights_only=True)
            torch.testing.assert_close(replaced["value"], torch.tensor([3.0]))
            self.assertEqual(list(Path(temporary).glob(".*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
