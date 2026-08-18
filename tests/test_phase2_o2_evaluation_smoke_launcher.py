from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_o2_evaluation_smoke.sbatch"


def test_evaluation_smoke_is_bounded_rocky9_gpu_and_provenance_locked():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --time=01:00:00" in source
    assert '"${VERSION_ID%%.*}" != "9"' in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "phase2_o2_scientific_evaluation_85604.json" in source
    assert "phase2_o2_training_freeze_6895637.json" in source
    assert "dd8951e39e60d1631866ebe7af7c4d529ad543daf211233369b8fec9936ee837" in source


def test_evaluation_smoke_runs_truth_separated_references_transport_and_one_checkpoint():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert '"${BUILD_REFERENCES}"' in source
    assert '"${EVALUATE_CHECKPOINT}"' in source
    assert source.count("--mode smoke") == 2
    assert "--arm C5P-H1" in source
    assert "--seed 1701" in source
    assert "--native-truth-result" in source
    assert "--geometry-manifest" in source
    assert 'references["target_frames"] != [498, 502]' in source
    assert 'set(references["references"])' in source
    assert 'candidate["gate"] is not None' in source
    assert 'candidate_score["target_truth_used_during_forecast_generation"] is not False' in source


def test_evaluation_smoke_runs_complete_suite_and_online_wandb_but_opens_no_gate():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert '"${PYTHON}" -m pytest -p no:cacheprovider -q' in source
    assert "WANDB_MODE=online" in source
    assert 'WANDB_DIR="${JOB_ROOT}/wandb_runtime"' in source
    assert "wandb.Api(timeout=30)" in source
    assert "wandb.init(" in source
    assert 'job_type="evaluation-smoke"' in source
    assert 'str(remote.state) != "finished"' in source
    assert '"scientific_authority": False' in source
    assert '"O2_scientific_gate_evaluated": False' in source
    assert '"O3_launch_allowed": False' in source
    assert "--mode full" not in source
    assert "/85606/" not in source
