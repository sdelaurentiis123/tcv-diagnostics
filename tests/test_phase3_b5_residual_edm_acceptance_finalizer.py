"""Contract tests for the no-rescore B5 one-seed gate finalizer."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "paper0/tools/finalize_b5_residual_edm_one_seed.py"


def load_finalizer():
    spec = importlib.util.spec_from_file_location(
        "finalize_b5_residual_edm_one_seed", FINALIZER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_B5_finalizer_summarizes_every_required_family_without_reduction() -> None:
    module = load_finalizer()
    acceptance = {
        "families": {
            name: {
                "passes": passes,
                "check_count": 10,
                "failed_check_count": 0 if passes else 2,
                "blocks_passing": 6 if passes else 4,
                "blocks_required": 5,
            }
            for name, passes in (
                ("field", True),
                ("spectral", False),
                ("transport", True),
            )
        }
    }
    summary = module._family_summary(acceptance)
    assert tuple(summary) == ("field", "spectral", "transport")
    assert summary["spectral"]["failed_check_count"] == 2
    assert summary["transport"]["blocks_required"] == 5


def test_B5_finalizer_locks_authorities_and_cannot_launch_downstream_work() -> None:
    module = load_finalizer()
    assert module.B5_MANIFEST_SHA256 == (
        "61f1fa565e2bcff008cbe72909daa97362dabe96d160a9beee4a3d5aa87d1334"
    )
    assert module.B5_COMPARATOR_SHA256 == (
        "2b04c10971e6d38ee439e33aa0b5331305acf16b38a96e7952fb26046049b5d2"
    )
    assert module.EXPECTED_B3_MANIFEST_SHA256 == (
        "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2"
    )
    source = FINALIZER.read_text(encoding="utf-8")
    assert '"O3_launch_allowed": False' in source
    assert '"additional_seed_training_authorized": False' in source
    assert '"held_out_85606_access_allowed": False' in source
    assert '"assimilation_allowed": False' in source
    assert '"diagnostic_ranking_allowed": False' in source


def test_B5_finalizer_is_a_pure_stored_artifact_reducer() -> None:
    source = FINALIZER.read_text(encoding="utf-8")
    assert "evaluate_b5_one_seed_acceptance" in source
    assert '"raw_forecast_changed": False' in source
    assert '"raw_score_changed": False' in source
    assert '"metrics_recomputed": False' in source
    assert '"training_performed": False' in source
    assert '"inference_performed": False' in source
    assert '"truth_scoring_performed": False' in source
    assert "score_b5_forecast" not in source
    assert "generate_selected_b5_forecasts" not in source
    assert "train_b5_edm_full" not in source
