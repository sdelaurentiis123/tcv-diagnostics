from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b2_finalize_event_eligibility_amendment.sbatch"


def test_A016_launcher_is_CPU_only_Rocky9_and_hash_locked() -> None:
    text = LAUNCHER.read_text()
    assert "#SBATCH --partition=gen" in text
    assert "#SBATCH --qos=gen" in text
    assert "#SBATCH --gres" not in text
    assert "VERSION_ID%%.*" in text
    assert '!= "9"' in text
    assert "PAPER0_EXPECTED_COMMIT" in text
    assert "Paper 0 checkout is dirty" in text
    assert "sha256sum -c \"${ORIGINAL_ROOT}/artifact_sha256.txt\"" in text
    assert "cd5d3a22b1a5f665c493417c3ea47bc7fd21d731e116f35a6a84eae68b462fd6" in text
    assert "6bb5d825b30c9c8292cda020d3bec824d9b04198617dc89afafa264daab44ea5" in text


def test_A016_launcher_is_gate_only_and_fail_closed() -> None:
    text = LAUNCHER.read_text()
    assert "training_performed=false" in text
    assert "inference_performed=false" in text
    assert "truth_scoring_performed=false" in text
    assert "held_out_85606_read=false" in text
    assert "raw_forecasts_changed" in text
    assert "raw_scores_changed" in text
    assert "metrics_recomputed" in text
    assert "O3_launch_allowed" in text
    assert "assimilation_allowed" in text
    assert "diagnostic_ranking_allowed" in text
    assert "Refusing to overwrite A016 result" in text
