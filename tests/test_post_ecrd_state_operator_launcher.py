"""Static safety checks for the bounded codec-free operator launch."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_launcher_is_small_bounded_and_gpu_only() -> None:
    text = (
        ROOT / "cluster/post_ecrd_codec_free_operator_smoke.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=16G" in text
    assert "#SBATCH --time=00:10:00" in text
    assert "PAPER0_EXPECTED_COMMIT" in text
    assert "WANDB_MODE=online" in text
    assert "--constraint=" not in text


def test_entrypoint_disables_tf32_without_relaxing_frozen_gate() -> None:
    text = (ROOT / "paper0/tools/smoke_codec_free_operator.py").read_text(
        encoding="utf-8"
    )
    assert "torch.backends.cudnn.allow_tf32 = False" in text
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in text
    assert 'torch.set_float32_matmul_precision("highest")' in text
    assert '"volume_normalized_maximum_absolute_error_max": 1.0e-3' in text
    assert '"volume_normalized_root_mean_square_error_max": 1.0e-4' in text
    assert '"boundary_normalized_maximum_absolute_error_max": 1.0e-4' in text
    assert '"boundary_normalized_root_mean_square_error_max": 1.0e-5' in text

