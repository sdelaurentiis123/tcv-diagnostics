"""Network-free tests for required B3 FGN evaluation W&B wrapping."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from paper0.tools.run_b3_fgn_evaluation_wandb import (
    evaluation_metrics,
    validate_evaluation_command,
)
from tcv_diagnostics.codec_training import sha256_path


def _args(tmp_path: Path) -> Namespace:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("print('ok')\n")
    output = tmp_path / "output"
    return Namespace(
        evaluator=evaluator,
        evaluator_sha256=sha256_path(evaluator),
        output_directory=output,
        paper0_commit="a" * 40,
        seed=1701,
        evaluation_args=[
            "--",
            "--mode",
            "smoke",
            "--seed",
            "1701",
            "--output-directory",
            str(output),
            "--paper0-commit",
            "a" * 40,
        ],
    )


def test_wrapper_locks_b3_identity_mode_and_heldout_paths(tmp_path: Path) -> None:
    args = _args(tmp_path)
    command, mode = validate_evaluation_command(args)
    assert mode == "smoke"
    assert command[-len(args.evaluation_args[1:]) :] == args.evaluation_args[1:]

    args.evaluation_args.append("/tmp/85606_forbidden")
    with pytest.raises(ValueError, match="held-out"):
        validate_evaluation_command(args)


def test_wrapper_extracts_compact_authoritative_metrics() -> None:
    aggregate = {
        "equal_channel_ensemble_mean_rmse": 1.0,
        "equal_channel_ensemble_mean_mae": 0.8,
        "equal_channel_fair_crps": 0.6,
        "equal_channel_corrected_spread_skill_ratio": 0.9,
        "all_fields_nonzero_spread": True,
    }
    result = {"held_out_85606_read": False}
    generation = {
        "target_count": 126,
        "forecast": {"shape": [126, 32, 1, 5, 64, 32, 88]},
        "wall_seconds": 4.0,
        "peak_cuda_memory_bytes": 1024,
    }
    score = {
        "field_and_marginal_calibration": {
            "regions": {"eligible_union": {"aggregate": aggregate}}
        }
    }
    metrics = evaluation_metrics(result, generation, score)
    assert metrics["forecast/ensemble_size"] == 32
    assert metrics["field/equal_channel_fair_crps"] == 0.6
    assert metrics["scope/held_out_85606_read"] is False
