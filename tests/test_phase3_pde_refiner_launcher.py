"""Static locks for the bounded Rocky 9 B4 PDE-Refiner smoke launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b4_pde_refiner_gpu_smoke.sbatch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_launcher_is_one_gpu_bounded_smoke_only_and_nondestructive() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --qos=gpupreempt" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --constraint=h100|h200" in source
    assert "#SBATCH --time=02:00:00" in source
    assert "#SBATCH --no-requeue" in source
    assert "--mode smoke" in source
    assert "--seed 1701" in source
    assert "train_b4_pde_refiner.py" in source
    assert "train_pde_refiner_full" not in source
    assert "--array" not in source
    assert "rm -" not in source
    assert "git reset" not in source
    assert "git checkout" not in source
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_launcher_requires_clean_commit_rocky9_hopper_wandb_and_full_suite() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "status --porcelain --untracked-files=all" in source
    assert '"${ID}" != "rocky"' in source
    assert '"${VERSION_ID%%.*}" != "9"' in source
    assert 'grep -Evq "H100|H200"' in source
    assert "export WANDB_MODE=online" in source
    assert "export WANDB_REQUIRE_SERVICE=true" in source
    assert '"${PYTHON}" -m pytest -p no:cacheprovider -q' in source
    assert 'tracking["remote_state_after_finish"] != "finished"' in source
    assert 'tracking["epochs_logged"] != 2' in source


def test_launcher_checks_identity_levels_stages_precision_and_scope() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    required = (
        'result["preoptimization_parent_identity"]["bitwise_exact"]',
        'result["deterministic_parent_load_audit"]["passed"]',
        'result["checkpoint_reload_bitwise_exact"]',
        'result["codec_bitwise_unchanged"]',
        'result["parent_parameter_gradient_seen"]',
        'result["refinement_parameter_gradient_seen"]',
        'result["training_level_counts"] != [8, 10, 7, 7]',
        'probe["canonical_stage_shape"] != [4, 2, 4, 5, 64, 32, 88]',
        'probe["level0_shared_bitwise_across_members"]',
        'probe["nonzero_final_diversity_in_every_field"]',
        'result["network_calls_per_member"] != 4',
        'result["training_dtype"] != "float32"',
        'result["torch_float32_matmul_precision"] != "highest"',
        'result["cuda_matmul_allow_tf32"] is not False',
        '"held_out_85606_read"',
        '"scientific_result"',
        '"full_B4_training_authorized"',
        '"H_det_evaluated"',
        '"H_prob_evaluated"',
        '"assimilation_allowed"',
        '"diagnostic_ranking_allowed"',
    )
    for text in required:
        assert text in source


def test_launcher_stages_only_verified_85604_model_data() -> None:
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


def test_launcher_embedded_local_source_hashes_match() -> None:
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
        "src/tcv_diagnostics/wandb_tracking.py": "wandb_tracking.py",
        "src/tcv_diagnostics/pde_refiner_wandb_tracking.py": (
            "pde_refiner_wandb_tracking.py"
        ),
    }
    for relative, suffix in mapping.items():
        pattern = re.compile(
            rf'check_sha256 "([0-9a-f]{{64}})" "\$\{{PAPER0_ROOT\}}/src/tcv_diagnostics/{re.escape(suffix)}"'
        )
        match = pattern.search(source)
        assert match is not None, relative
        assert match.group(1) == sha256(ROOT / relative), relative

    entry_match = re.search(
        r'check_sha256 "([0-9a-f]{64})" "\$\{ENTRYPOINT\}"', source
    )
    assert entry_match is not None
    assert entry_match.group(1) == sha256(
        ROOT / "paper0/tools/train_b4_pde_refiner.py"
    )


def test_launcher_locks_protocol_manifest_parent_codec_and_normalization() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert sha256(ROOT / "paper0/protocol/PHASE3_B4_PDE_REFINER_PROTOCOL.md") in source
    assert sha256(ROOT / "paper0/manifests/phase3_b4_pde_refiner_85604.json") in source
    assert sha256(ROOT / "paper0/results/phase3_b3_fgn_one_seed_gate_6899224.json") in source
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


def test_launcher_indexes_every_required_smoke_artifact() -> None:
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
        "validation_decoded_stages.pt",
        "wandb.json",
        "artifact_sha256.txt",
    ):
        assert name in source
