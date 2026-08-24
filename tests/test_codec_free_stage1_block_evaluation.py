"""Known-answer tests for the Stage-1 chronological-block evaluator."""

from __future__ import annotations

from pathlib import Path

from tcv_diagnostics.models.codec_free_operator import CodecFreeOperatorConfig


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0/tools/evaluate_codec_free_stage1_blocks.py"


def load_namespace():
    namespace = {"__name__": "stage1_block_evaluation_test"}
    exec(compile(TOOL.read_text(encoding="utf-8"), str(TOOL), "exec"), namespace)
    return namespace


def test_primary_blocks_are_frozen_and_matched() -> None:
    blocks = load_namespace()["BLOCKS"]
    assert blocks == {
        "V00": (498, 540),
        "V01": (540, 582),
        "V02": (582, 624),
    }
    assert all(stop - start == 42 for start, stop in blocks.values())


def test_restore_config_ignores_only_derived_record_fields() -> None:
    restore = load_namespace()["restore_config"]
    source = CodecFreeOperatorConfig(
        state_family="c5p",
        predict_boundary=False,
        zero_initialize_output=True,
    )
    restored = restore(source.to_record())
    assert restored == source


def test_evaluator_fails_closed_on_held_out_and_physics_flags() -> None:
    text = TOOL.read_text(encoding="utf-8")
    assert '"held_out_85606_read": False' in text
    assert '"physics_derived_loss_used": False' in text
    assert "requires an allocated CUDA GPU" in text
