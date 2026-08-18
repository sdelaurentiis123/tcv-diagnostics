from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_o2_evaluation_full.sbatch"


def test_full_evaluation_uses_exact_clean_rocky9_four_h200_job():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpuxl" in source
    assert "#SBATCH --gres=gpu:h200:4" in source
    assert "#SBATCH --constraint=h200" in source
    assert '"${VERSION_ID%%.*}" != "9"' in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "phase2_o2_evaluation_smoke_6895931.json" in source
    assert "dc53d9561d0ef0f00cbb41eb14f510fd4a19cf5427acd17be2947baff4211273" in source


def test_full_evaluation_is_the_exact_six_seed_matrix_with_frozen_references():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert source.count("--mode full") == 3
    assert 'arms=("C5P-H1" "C5P-H1" "C5P-H1" "C5P-H2" "C5P-H2" "C5P-H2")' in source
    assert "seeds=(1701 1702 1703 1701 1702 1703)" in source
    assert "Building the frozen full references once" in source
    assert 'for run_index in 0 1 2 3 4 5' in source
    assert "Launching full O2 evaluation wave 0" in source
    assert "Launching full O2 evaluation wave 1" in source
    assert '"${FINALIZE_MATRIX}"' in source
    assert 'matrix["run_count"] != 6' in source
    assert '"target_frames": [498, 624]' in source
    for digest in (
        "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e",
        "d15c74717fad6a3ccf5b5af895e3eefb7271667f4bbde2164514a61a526bc0e8",
        "a718b2135c7019d05541bd5ffb029ce9408df8225603cffc957c42d2ce5abae3",
        "3b971b2081901469e1f98adbe27b5cdbf3281d08a99ee28e0d8d8b1577722a84",
        "5edc3e002730eb78232967255cfab66ee860b8b3858eed007f7061341b5c36eb",
        "a70bd271117f1b0afb21258e4c5d7d4eb4919dc4a528509ccbf6ac2464622d85",
    ):
        assert digest in source


def test_full_evaluation_tracks_wandb_and_cannot_advance_protocol_automatically():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert '"${PYTHON}" -m pytest -p no:cacheprovider -q' in source
    assert "WANDB_MODE=online" in source
    assert 'WANDB_DIR="${JOB_ROOT}/wandb_runtime"' in source
    assert "wandb.init(" in source
    assert 'job_type="scientific-evaluation"' in source
    assert 'str(remote.state) != "finished"' in source
    assert 'matrix["O3_launch_allowed"] is not False' in source
    assert '"O3_launch_allowed": False' in source
    assert '"stochastic_model_authorized": False' in source
    assert '"held_out_85606_access_allowed": False' in source
    assert "/85606/" not in source
