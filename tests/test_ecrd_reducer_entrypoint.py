"""Contract tests for the data-free ECRD model-ladder reducer."""

from __future__ import annotations

from pathlib import Path

import pytest

from paper0.tools.summarize_ecrd_model_ladder import (
    parse_score_specifications,
)
from tcv_diagnostics.ecrd_training import ECRD_ARMS, ECRD_MODEL_SEEDS


ROOT = Path(__file__).resolve().parents[1]


def _specifications() -> list[str]:
    return [
        f"{arm}:{seed}:/tmp/{arm}_{seed}.json:{seed:064x}"
        for arm in ECRD_ARMS
        for seed in ECRD_MODEL_SEEDS
    ]


def test_score_specifications_require_the_complete_matched_matrix() -> None:
    matrix = parse_score_specifications(_specifications())
    assert tuple(matrix) == ECRD_ARMS
    for arm in ECRD_ARMS:
        assert tuple(sorted(matrix[arm])) == ECRD_MODEL_SEEDS
        for seed in ECRD_MODEL_SEEDS:
            assert matrix[arm][seed][0] == Path(f"/tmp/{arm}_{seed}.json")


def test_score_specifications_reject_duplicates_and_missing_runs() -> None:
    specifications = _specifications()
    with pytest.raises(ValueError, match="complete"):
        parse_score_specifications(specifications[:-1])
    with pytest.raises(ValueError, match="duplicated"):
        parse_score_specifications(specifications + [specifications[0]])


def test_reducer_is_data_free_and_cannot_authorize_the_holdout() -> None:
    source = (ROOT / "paper0/tools/summarize_ecrd_model_ladder.py").read_text()
    assert "load_official_catalog" not in source
    assert "NativeTruthCatalog" not in source
    assert '"held_out_85606_access_authorized": False' in source
    assert "evaluate_ecrd_model_ladder" in source
