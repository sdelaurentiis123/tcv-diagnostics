from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "paper0/tools/build_o2_references.py"
EVALUATE = ROOT / "paper0/tools/evaluate_o2_checkpoint.py"
FINALIZE = ROOT / "paper0/tools/finalize_o2_evaluation.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", [REFERENCE, EVALUATE, FINALIZE])
def test_o2_evaluation_entrypoints_parse_help(path: Path):
    completed = subprocess.run(
        [sys.executable, str(path), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert "85606" not in completed.stdout


def test_checkpoint_authorization_matches_one_exact_frozen_arm_seed(tmp_path: Path):
    module = _load(EVALUATE, "paper0_evaluate_o2_checkpoint")
    checkpoint = tmp_path / "selected.pt"
    codec = tmp_path / "codec.pt"
    checkpoint.write_bytes(b"selected")
    codec.write_bytes(b"codec")
    run = {
        "run_index": 4,
        "arm": "C5P-H2",
        "seed": 1702,
        "selected_checkpoint": {
            "path": str(checkpoint),
            "sha256": "a" * 64,
        },
        "codec_checkpoint": {
            "path": str(codec),
            "sha256": "b" * 64,
            "trainable_during_O2": False,
        },
    }
    freeze = {
        "scope": "phase2_C5P_O2_full_training_matrix_frozen",
        "status": "completed_pending_scientific_O2_evaluation",
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_commit": "c" * 40,
        "checkpoint_choice_frozen_before_reference_or_physics_metrics": True,
        "O2_scientific_evaluation_completed": False,
        "O3_launch_allowed": False,
        "runs": [run],
    }
    selected = module.frozen_run(
        freeze,
        arm="C5P-H2",
        seed=1702,
        checkpoint=checkpoint,
        checkpoint_sha256="a" * 64,
        codec_checkpoint=codec,
        codec_checkpoint_sha256="b" * 64,
        training_commit="c" * 40,
    )
    assert selected["run_index"] == 4
    with pytest.raises(RuntimeError, match="artifacts differ"):
        module.frozen_run(
            freeze,
            arm="C5P-H2",
            seed=1702,
            checkpoint=checkpoint,
            checkpoint_sha256="0" * 64,
            codec_checkpoint=codec,
            codec_checkpoint_sha256="b" * 64,
            training_commit="c" * 40,
        )


def test_forecast_and_score_are_separate_and_next_stage_remains_closed():
    reference = REFERENCE.read_text(encoding="utf-8")
    evaluate = EVALUATE.read_text(encoding="utf-8")
    finalize = FINALIZE.read_text(encoding="utf-8")
    assert "OneStepContextDataset" in evaluate
    assert "generate_selected_o2_forecasts(" in evaluate
    assert "score_o2_forecast(" in evaluate
    assert evaluate.index("generate_selected_o2_forecasts(") < evaluate.index(
        "score_o2_forecast("
    )
    assert '"target_truth_read": False' in evaluate
    assert "fit_training_only_o2_ar1(" in reference
    assert '"validation_tuning_used": False' in reference
    assert '"O3_launch_allowed": False' in reference
    assert '"O3_launch_allowed": False' in evaluate
    assert '"O3_launch_allowed": False' in finalize
    assert '"held_out_85606_access_allowed": False' in finalize
