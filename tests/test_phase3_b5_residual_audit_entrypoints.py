"""Network-free B5 residual-audit entrypoint and W&B-wrapper tests."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import json

import pytest

from paper0.tools.run_b5_residual_audit_wandb import (
    audit_metrics,
    validate_audit_command,
)
from paper0.tools.audit_b5_h1_training_residual import (
    _decorrelation_frames,
    _verify_audit_authority,
    _verify_b4_stop_gate,
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
        evaluation_args=[
            "--",
            "--output-directory",
            str(output),
            "--paper0-commit",
            "a" * 40,
        ],
    )


def test_b5_wandb_wrapper_locks_command_and_heldout_paths(tmp_path: Path) -> None:
    args = _args(tmp_path)
    command = validate_audit_command(args)
    assert command[-len(args.evaluation_args[1:]) :] == args.evaluation_args[1:]
    args.evaluation_args.append("/tmp/85606_forbidden")
    with pytest.raises(ValueError, match="held-out"):
        validate_audit_command(args)


def test_b5_wandb_metric_projection_is_compact_and_scope_explicit() -> None:
    result = {
        "target_count": 430,
        "wall_seconds": 10.0,
        "generation": {"wall_seconds": 5.0, "peak_cuda_memory_bytes": 1024},
        "held_out_85606_read": False,
        "validation_frames_read": False,
        "training_performed": False,
        "B5_training_authorized": False,
        "fields": ["Ne"],
    }
    audit = {
        "cross_field": {"global": {"entropy_effective_rank": 2.5}},
        "scale": {
            "global": {
                "Ne": {
                    "RMS": 0.2,
                    "MAE": 0.1,
                    "bias": -0.01,
                    "residual_to_target_variance_ratio": 0.3,
                }
            },
            "heteroscedasticity": {"Ne": {"q95_to_q05_ratio": 4.0}},
        },
        "temporal_autocorrelation": {
            "pattern": {
                "fields": {
                    "Ne": {"length_summary": {"first_stable_near_zero_lag": 7}}
                }
            }
        },
        "toroidal_support": {
            "fields": {
                "Ne": {
                    "bands": {"k1_3": {"residual_power_fraction": 0.4}}
                }
            }
        },
    }
    metrics = audit_metrics(result, audit)
    assert metrics["residual/Ne/RMS"] == 0.2
    assert metrics["temporal/Ne/stable_near_zero_frames"] == 7
    assert metrics["scope/held_out_85606_read"] is False
    assert metrics["scope/B5_training_authorized"] is False


def test_b5_evaluator_source_makes_truth_opening_order_explicit() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "paper0/tools/audit_b5_h1_training_residual.py"
    ).read_text()
    assert source.index("generate_frozen_h1_training_forecast") < source.index(
        "truth_dataset = CodecFrameDataset"
    )
    assert source.index("forecast_sha256 = generation") < source.index(
        "truth_dataset = CodecFrameDataset"
    )
    assert '"B5_training_authorized": False' in source
    assert '"held_out_85606_read": False' in source


def test_b5_evaluator_accepts_exact_frozen_authorities() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "paper0/manifests/phase3_b5_residual_audit_85604.json").read_text()
    )
    parent = manifest["deterministic_mean"]
    codec = manifest["codec"]
    args = Namespace(
        checkpoint=Path(parent["checkpoint_path"]),
        checkpoint_sha256=parent["checkpoint_sha256"],
        codec_checkpoint=Path(codec["checkpoint_path"]),
        codec_checkpoint_sha256=codec["checkpoint_sha256"],
        latent_normalization=Path(codec["latent_normalization_path"]),
        latent_normalization_sha256=codec["latent_normalization_sha256"],
        training_commit=parent["training_commit"],
    )
    _verify_audit_authority(manifest, args=args)
    decorrelation = json.loads(
        (root / "paper0/results/phase1_85604_profile_6890606.json").read_text()
    )
    assert _decorrelation_frames(decorrelation, manifest) == 2.2443947105846638
    b4 = json.loads(
        (
            root
            / "paper0/results/phase3_b4_pde_refiner_one_seed_gate_6901285.json"
        ).read_text()
    )
    _verify_b4_stop_gate(b4)
