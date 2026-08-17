"""Known-answer tests for matched O1 checkpoint and state-view contracts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tcv_diagnostics.codec_training import CodecRunConfig, sha256_path
from tcv_diagnostics.matched_codec_evaluation import (
    COMMON_FIELDS,
    derive_e6b_common_components,
    derive_e6b_common_physical,
    native81_candidate_fields,
    native_view_spec,
    validate_selected_checkpoint,
)
from tcv_diagnostics.model_training_data import VOLUME_SHAPE


class TestMatchedCodecEvaluation(unittest.TestCase):
    def test_checkpoint_contract_accepts_only_completed_selected_run(self) -> None:
        config = CodecRunConfig.frozen(
            mode="full", codec="dcae_l20", family="c5p", seed=1701
        )
        epoch = 12
        loss = 0.031
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "selected.pt"
            torch.save({"placeholder": True}, checkpoint)
            checkpoint_hash = sha256_path(checkpoint)
            result = {
                "scope": "O1_codec_full",
                "held_out_85606_read": False,
                "development_run": "85604",
                "completed_epochs": 200,
                "physics_derived_loss_used": False,
                "selected_checkpoint": {"sha256": checkpoint_hash},
                "config": config.to_record(),
                "paper0_commit": "abc123",
                "selected_epoch": epoch,
                "selected_validation_equal_channel_mae": loss,
                "checkpoint_reload_bitwise_exact": True,
            }
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            payload = {
                "kind": "selected_model",
                "config": config.to_record(),
                "model_state": {},
                "reload_probe": {},
                "paper0_commit": "abc123",
                "epoch": epoch,
                "global_step": (epoch + 1) * config.optimizer_steps_per_epoch,
                "validation_loss": loss,
            }
            identity = validate_selected_checkpoint(
                checkpoint_path=checkpoint,
                checkpoint_sha256=checkpoint_hash,
                payload=payload,
                training_result_path=result_path,
                training_result_sha256=sha256_path(result_path),
                training_result=result,
                codec="dcae_l20",
                family="c5p",
                seed=1701,
            )
            self.assertEqual(identity.selected_epoch, epoch)

            payload["optimizer_state"] = {}
            with self.assertRaisesRegex(ValueError, "optimizer"):
                validate_selected_checkpoint(
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=checkpoint_hash,
                    payload=payload,
                    training_result_path=result_path,
                    training_result_sha256=sha256_path(result_path),
                    training_result=result,
                    codec="dcae_l20",
                    family="c5p",
                    seed=1701,
                )

    def test_e6b_common_view_uses_deuterium_momentum_identity(self) -> None:
        state = np.ones((1, 6, *VOLUME_SHAPE), dtype=np.float32)
        state[:, 0] = 2.0
        state[:, 1] = 3.0
        state[:, 2] = 4.0
        state[:, 4] = 12.0
        phi = np.full((1, *VOLUME_SHAPE), 5.0, dtype=np.float32)
        common = derive_e6b_common_physical(state, phi)
        self.assertEqual(tuple(common.shape), (1, 5, *VOLUME_SHAPE))
        self.assertEqual(COMMON_FIELDS, ("Ne", "Pe", "Pi", "phi", "Vi"))
        self.assertEqual(float(common[0, 4, 0, 0, 0]), 3.0)

        state[:, 0, 0, 0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "non-positive"):
            derive_e6b_common_physical(state, phi)

    def test_e6b_common_components_need_no_unused_e6b_channels(self) -> None:
        shape = (2, *VOLUME_SHAPE)
        common = derive_e6b_common_components(
            ne=np.full(shape, 2.0, dtype=np.float32),
            pe=np.full(shape, 3.0, dtype=np.float32),
            pi=np.full(shape, 4.0, dtype=np.float32),
            nvi=np.full(shape, 12.0, dtype=np.float32),
            phi=np.full(shape, 5.0, dtype=np.float32),
        )
        self.assertEqual(common.shape, (2, 5, *VOLUME_SHAPE))
        np.testing.assert_array_equal(common[:, 4], 3.0)

    def test_native81_fields_follow_family_contract(self) -> None:
        state = np.ones((1, 5, *VOLUME_SHAPE), dtype=np.float32)
        state[:, 0] = 2.0
        native = native81_candidate_fields("c5p", state)
        self.assertEqual(set(native), {"Ne", "Pe", "Pi", "phi"})
        self.assertEqual(tuple(native["phi"].shape), (1, 64, 32, 81))
        self.assertEqual(native["phi"].dtype, np.dtype("f4"))
        self.assertEqual(native_view_spec("c5p").name, "c5p_native_and_common")


if __name__ == "__main__":
    unittest.main()
