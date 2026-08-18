"""Static regression checks for the one-seed full B3 Slurm launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b3_fgn_full.sbatch"


def test_full_launcher_shell_syntax_and_single_gpu_scope() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --qos=gpupreempt" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --constraint=h100|h200" in source
    assert "#SBATCH --time=24:00:00" in source
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --array" not in source
    assert "--mode full" in source
    assert "--seed 1701" in source
    assert "train_b3_fgn_full.py" in source
    assert "train_b3_fgn.py\"" not in source


def test_full_launcher_locks_protocol_evidence_and_training_sources() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    required_hashes = (
        "db717c5605ad9653d2b051ec13254b43bf230f514cb173d295e95d3c68af8030",
        "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2",
        "c31d24d843050c5708440018217e48ca66f5c3a3f4ee0ddb110a3287a73292d1",
        "dbac54c033917abbfec7e380d96a0c9be93667ae58240b4403400b57c76e2808",
        "251820a6f81d97ffdb046eba7a23cd12505c1179e600e7942231de2fd1feeacb",
        "15bcfc0b4c9cec2a858848e1d8fbc0fdf2da6f8a3bc6fd001e5c233abae7b397",
        "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e",
        "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d",
        "afcb0eda5d611d58f6eb2340aa55cfecd1a231b83a6912d9db398be706296738",
        "69c5e140e6f456e2442ae40ec5be7c04c5d228baf6ec95e119de5a3d1c86e79d",
        "1a8ec4227f5de595f27e9d2679ca3b830e598516d56e19d01910cc1c0413fef6",
        "10529f5be61306c8ceb50314075a4a678254cb553cf9638e368e4e0ea9be29b5",
        "acb20cc56ee277e5b2c18ebfa0c435d56354394c6eddd3cff452c6be0040e77d",
    )
    for digest in required_hashes:
        assert digest in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "status --porcelain --untracked-files=all" in source


def test_full_launcher_requires_tests_wandb_staging_and_postconditions() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "WANDB_MODE=online" in source
    assert "wandb_preflight.json" in source
    assert '"${PYTHON}" -m pytest -p no:cacheprovider -q' in source
    assert "verified_shards" in source
    assert 'if [[ "${verified_shards}" -ne 8 ]]' in source
    assert 'result["completed_epochs"] != 100' in source
    assert 'result["completed_optimizer_steps"] != 2700' in source
    assert 'tracking["epochs_logged"] != 100' in source
    assert 'result["training_complete_is_scientific_acceptance"] is not False' in source
    assert '"probabilistic_scientific_gate_evaluated",' in source
    assert "B3 remains scientifically undecided" in source


def test_full_launcher_is_non_overwriting_and_85604_only() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'if [[ -e "${JOB_ROOT}" ]]' in source
    assert 'if [[ -e "${NODE_LOCAL_ROOT}" ]]' in source
    assert "Refusing to overwrite" in source
    assert "phase2_model_dataset/job_6893525" in source
    assert "phase2_model_dataset_85604" in source
    assert "/85606/" not in source
    assert "--artifact-root \"${MODEL_DATA_STAGED}\"" in source
