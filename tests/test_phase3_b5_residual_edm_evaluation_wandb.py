"""Tests for the required B5 evaluation W&B wrapper."""

from __future__ import annotations

from argparse import Namespace
import importlib.util
from pathlib import Path

import pytest

from tcv_diagnostics.codec_training import sha256_path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "paper0/tools/run_b5_residual_edm_evaluation_wandb.py"


def load_wrapper():
    spec = importlib.util.spec_from_file_location(
        "run_b5_residual_edm_evaluation_wandb", WRAPPER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(tmp_path: Path) -> Namespace:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("print('ok')\n", encoding="utf-8")
    output = tmp_path / "evaluation"
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


def test_B5_wandb_wrapper_locks_identity_mode_and_heldout_paths(
    tmp_path: Path,
) -> None:
    module = load_wrapper()
    values = args(tmp_path)
    command, mode = module.validate_evaluation_command(values)
    assert mode == "smoke"
    assert command[-len(values.evaluation_args[1:]) :] == values.evaluation_args[1:]

    values.evaluation_args.append("/tmp/85606_forbidden")
    with pytest.raises(ValueError, match="held-out"):
        module.validate_evaluation_command(values)


def test_B5_wandb_wrapper_rejects_hash_or_duplicate_identity(tmp_path: Path) -> None:
    module = load_wrapper()
    values = args(tmp_path)
    values.evaluator_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        module.validate_evaluation_command(values)

    values = args(tmp_path)
    values.evaluation_args.extend(("--seed", "1701"))
    with pytest.raises(ValueError, match="exactly one --seed"):
        module.validate_evaluation_command(values)


def test_B5_wandb_wrapper_extracts_compact_authoritative_metrics() -> None:
    module = load_wrapper()
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
        "inference": {"sampler_steps": 18, "network_evaluations_per_member": 35},
    }
    score = {
        "field_and_marginal_calibration": {
            "regions": {"eligible_union": {"aggregate": aggregate}}
        }
    }
    metrics = module.evaluation_metrics(result, generation, score)
    assert metrics["forecast/ensemble_size"] == 32
    assert metrics["forecast/network_evaluations_per_member"] == 35
    assert metrics["field/equal_channel_fair_crps"] == 0.6
    assert metrics["scope/held_out_85606_read"] is False


def test_B5_wandb_wrapper_requires_online_remote_verification_in_source() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert 'mode="online"' in source
    assert 'resume="never"' in source
    assert "remote_presence_verified_after_finish" in source
    assert 'str(remote.state) != "finished"' in source
    assert '"forecasts_uploaded": False' in source
    assert '"local_artifacts_are_scientific_authority": True' in source
