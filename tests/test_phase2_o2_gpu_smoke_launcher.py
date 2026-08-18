import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_o2_gpu_smoke.sbatch"
LOCKED = (
    ROOT / "paper0/manifests/phase2_c5p_o2_continuation_85604.json",
    ROOT / "paper0/protocol/PHASE2_C5P_O2_CONTINUATION_PROTOCOL.md",
    ROOT / "paper0/results/phase2_matched_o1_finalize_r2_6894863.json",
    ROOT / "paper0/results/phase2_model_dataset_6893525.json",
    ROOT / "paper0/results/phase2_model_dataset_normalization_6893525.json",
    ROOT / "src/tcv_diagnostics/models/layers.py",
    ROOT / "src/tcv_diagnostics/models/dcae.py",
    ROOT / "src/tcv_diagnostics/models/__init__.py",
    ROOT / "src/tcv_diagnostics/models/vit.py",
    ROOT / "src/tcv_diagnostics/models/o2.py",
    ROOT / "src/tcv_diagnostics/model_training_data.py",
    ROOT / "src/tcv_diagnostics/o2_training_data.py",
    ROOT / "src/tcv_diagnostics/o2_training.py",
    ROOT / "src/tcv_diagnostics/codec_training.py",
    ROOT / "src/tcv_diagnostics/wandb_tracking.py",
    ROOT / "paper0/tools/train_o2.py",
)


def test_smoke_is_rocky9_one_h100_bounded_and_non_overwriting():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in text
    assert "#SBATCH --qos=gpupreempt" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --constraint=h100" in text
    assert '"${VERSION_ID%%.*}" != "9"' in text
    assert "--mode smoke" in text
    assert '--arm "${arm}"' in text
    assert 'run_smoke "C5P-H1" "c5p_h1"' in text
    assert 'run_smoke "C5P-H2" "c5p_h2"' in text
    assert "--seed 1701" in text
    assert "completed_epochs\"] != 2" in text
    assert "too many training targets" in text
    assert "Refusing to overwrite existing result directory" in text


def test_smoke_requires_clean_exact_commit_full_cpu_suite_and_online_wandb():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "PAPER0_EXPECTED_COMMIT" in text
    assert "status --porcelain --untracked-files=all" in text
    assert '"${PYTHON}" -m pytest -p no:cacheprovider -q' in text
    assert "WANDB_MODE=online" in text
    assert "wandb_preflight.json" in text
    assert 'tracking["remote_state_after_finish"] != "finished"' in text


def test_smoke_hash_locks_every_local_dependency_and_exact_codec():
    text = LAUNCHER.read_text(encoding="utf-8")
    for path in LOCKED:
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text
    assert "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d" in text


def test_smoke_cannot_claim_science_or_open_later_stages():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert '"scientific_result": False' in text
    assert '"O2_scientific_gate_evaluated": False' in text
    assert '"O3_launch_allowed": False' in text
    assert 'result["physics_derived_loss_used"] is not False' in text
    assert 'result["target_truth_used_as_model_input"] is not False' in text
    assert 'result["held_out_85606_read"] is not False' in text
