"""Contract tests for the frozen matched H1 comparator builder."""

from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path

import pytest

from paper0.tools.build_b3_h1_comparator import (
    build_h1_field_comparator,
    comparator_inputs_from_manifest,
    frozen_best_uncompressed,
)
from tcv_diagnostics.model_data import load_strict_json


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0/manifests/phase3_b3_full_evaluation_85604.json"
LAUNCHER = ROOT / "cluster/phase3_b3_h1_comparator.sbatch"


def test_manifest_selects_exact_H1_parent_and_preexisting_uncompressed_reference() -> None:
    manifest = load_strict_json(MANIFEST)
    parent, uncompressed = comparator_inputs_from_manifest(manifest)
    assert parent["arm"] == "C5P-H1"
    assert parent["seed"] == 1701
    assert parent["forecast_sha256"] == (
        "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
    )
    assert parent["score_sha256"] == (
        "ebdc707e2be500af7de492038ae8bfb4d126b81b271b340345b85a7fba1d5593"
    )
    assert uncompressed["names"] == [
        "persistence",
        "training_only_toroidal_spectral_AR1",
    ]

    contaminated = deepcopy(manifest)
    contaminated["comparators"]["primary_deterministic_parent"]["arm"] = "C5P-H2"
    with pytest.raises(ValueError, match="H1 comparator"):
        comparator_inputs_from_manifest(contaminated)


def test_reused_uncompressed_record_must_predate_B3_and_use_gauge_policy() -> None:
    record = {
        "scope": "phase3_B2_frozen_paired_deterministic_comparators_85604",
        "status": "completed_before_B2_scientific_acceptance",
        "development_run": "85604",
        "held_out_85606_read": False,
        "B2_forecasts_or_scores_read": False,
        "scientific_acceptance_evaluated": False,
        "best_uncompressed": {
            "name": "training_only_toroidal_spectral_AR1",
            "field": {
                "target_frames": [498, 624],
                "potential_policy": (
                    "subtract_full_spatial_mean_separately_per_forecast_and_truth_target"
                ),
            },
        },
    }
    assert frozen_best_uncompressed(record)["name"].endswith("spectral_AR1")
    record["best_uncompressed"]["name"] = "persistence"
    with pytest.raises(ValueError, match="uncompressed"):
        frozen_best_uncompressed(record)


def test_H1_field_builder_uses_one_context_and_relabels_legacy_accumulator() -> None:
    source = inspect.getsource(build_h1_field_comparator)
    assert "context_frames=1" in source
    assert 'record["scope"] = "gauge_consistent_deterministic_C5P_H1_field_comparator"' in source
    assert 'record["context_frames"] = 1' in source


def test_launcher_is_cpu_only_hash_locked_and_forbids_scientific_progression() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in text
    assert "#SBATCH --qos=gen" in text
    assert "#SBATCH --gres" not in text
    assert "PAPER0_EXPECTED_COMMIT" in text
    assert "phase3_b3_full_evaluation_85604.json" in text
    assert "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2" in text
    assert "2e96359cf2213d62ea81c7bec33e30551ade6fd081ca88a1aa088f65d84de72e" in text
    assert "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed" in text
    assert "ebdc707e2be500af7de492038ae8bfb4d126b81b271b340345b85a7fba1d5593" in text
    assert 'result["B3_forecasts_or_scores_read"] is not False' in text
    assert 'result["O3_launch_allowed"] is not False' in text
