"""Static fail-closed checks for the Rocky 9 full B4 launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b4_pde_refiner_full.sbatch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_launcher_is_one_gpu_full_seed1701_rusty9_only() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --constraint=h100|h200" in source
    assert "#SBATCH --time=24:00:00" in source
    assert '"${PYTHON}" -u "${ENTRYPOINT}"' in source
    assert "--mode full" in source
    assert "--seed 1701" in source
    assert "--mode smoke" not in source
    assert "train_b4_pde_refiner_full.py" in source
    assert "train_b4_pde_refiner.py" not in source
    assert "Full B4 training requires Rocky Linux 9" in source
    assert "H100|H200" in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "status --porcelain --untracked-files=all" in source


def test_launcher_keeps_training_non_scientific_and_later_phases_closed() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for text in (
        'result["training_complete_is_scientific_acceptance"] is not False',
        '"scientific_B4_evaluation_performed"',
        '"H_det_evaluated"',
        '"H_prob_evaluated"',
        '"O3_launch_allowed"',
        '"assimilation_allowed"',
        '"diagnostic_ranking_allowed"',
        '"held_out_85606_read"',
    ):
        assert text in source
    assert "B4 remains scientifically undecided" in source
    assert "/85606/" not in source


def test_launcher_hash_locks_protocol_manifest_and_passing_smoke() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for relative in (
        "paper0/protocol/PHASE3_B4_FULL_TRAINING_EVALUATION_PROTOCOL.md",
        "paper0/manifests/phase3_b4_full_evaluation_85604.json",
        "paper0/protocol/PHASE3_B4_PDE_REFINER_PROTOCOL.md",
        "paper0/manifests/phase3_b4_pde_refiner_85604.json",
        "paper0/results/phase3_b4_pde_refiner_gpu_smoke_6899469.json",
    ):
        assert sha256(ROOT / relative) in source, relative
    assert (
        'readonly PARENT_SHA="5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"'
        in source
    )
    assert (
        'readonly C5P_CODEC_SHA="9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"'
        in source
    )
    assert (
        'readonly LATENT_NORMALIZATION_SHA="afcb0eda5d611d58f6eb2340aa55cfecd1a231b83a6912d9db398be706296738"'
        in source
    )


def test_launcher_embedded_source_hashes_match_current_files() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    mapping = {
        "src/tcv_diagnostics/models/layers.py": "models/layers.py",
        "src/tcv_diagnostics/models/dcae.py": "models/dcae.py",
        "src/tcv_diagnostics/models/o2.py": "models/o2.py",
        "src/tcv_diagnostics/models/vit.py": "models/vit.py",
        "src/tcv_diagnostics/models/pde_refiner.py": "models/pde_refiner.py",
        "src/tcv_diagnostics/model_training_data.py": "model_training_data.py",
        "src/tcv_diagnostics/o2_training_data.py": "o2_training_data.py",
        "src/tcv_diagnostics/o2_training.py": "o2_training.py",
        "src/tcv_diagnostics/codec_training.py": "codec_training.py",
        "src/tcv_diagnostics/pde_refiner_training.py": "pde_refiner_training.py",
        "src/tcv_diagnostics/pde_refiner_full_training.py": (
            "pde_refiner_full_training.py"
        ),
        "src/tcv_diagnostics/wandb_tracking.py": "wandb_tracking.py",
        "src/tcv_diagnostics/pde_refiner_full_wandb_tracking.py": (
            "pde_refiner_full_wandb_tracking.py"
        ),
    }
    for relative, suffix in mapping.items():
        pattern = re.compile(
            rf'check_sha256 "([0-9a-f]{{64}})" "\$\{{PAPER0_ROOT\}}/src/tcv_diagnostics/{re.escape(suffix)}"'
        )
        match = pattern.search(source)
        assert match is not None, relative
        assert match.group(1) == sha256(ROOT / relative), relative
    entry = re.search(
        r'check_sha256 "([0-9a-f]{64})" "\$\{ENTRYPOINT\}"',
        source,
    )
    assert entry is not None
    assert entry.group(1) == sha256(
        ROOT / "paper0/tools/train_b4_pde_refiner_full.py"
    )


def test_launcher_stages_only_all_eight_verified_85604_shards() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "phase2_model_dataset/job_6893525" in source
    assert "phase2_model_dataset_85604" in source
    assert 'if [[ "${verified_shards}" -ne 8 ]]' in source
    assert (
        "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18"
        in source
    )
    assert (
        "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7"
        in source
    )
    assert (
        "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01"
        in source
    )


def test_launcher_validates_full_budget_schedule_precision_and_wandb() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for text in (
        'result["completed_epochs"] != 100',
        'result["completed_optimizer_steps"] != 2700',
        'result["EMA_updates"] != 2700',
        'result["validation_candidates_evaluated"] != 20',
        'list(range(5, 101, 5))',
        '[10831, 10680, 10722, 10767]',
        '[126, 2, 3]',
        'history[-1]["learning_rate"] != 1e-6',
        'tracking["epochs_logged"] != 100',
        'result["torch_float32_matmul_precision"] != "highest"',
        'result["cuda_matmul_allow_tf32"] is not False',
    ):
        assert text in source
    assert "WANDB_MODE=online" in source
    assert "wandb_preflight.json" in source
    assert "PDERefinerFullOnlineWandbTracker" not in source


def test_launcher_indexes_every_required_training_artifact() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for name in (
        "config.json",
        "latent_normalization.json",
        "training_levels.npy",
        "validation_seed_bank.npy",
        "history.jsonl",
        "result.json",
        "selected.pt",
        "final_training_state.pt",
        "wandb.json",
        "artifact_sha256.txt",
        "environment.txt",
        "slurm_job.txt",
        "test_output.txt",
        "wandb_preflight.json",
    ):
        assert name in source
